import argparse
import copy

import numpy as np
import pandas as pd
import torch

import flamby.strategies as strats
from flamby.benchmarks.benchmark_utils import (
    ensemble_perf_from_predictions,
    evaluate_model_on_local_and_pooled_tests,
    fill_df_with_xp_results,
    find_xps_in_df,
    get_logfile_name_from_strategy,
    init_data_loaders,
    init_xp_plan,
    set_dataset_specific_config,
    set_seed,
    train_single_centric,
)
from flamby.benchmarks.conf import (
    check_config,
    get_dataset_args,
    get_results_file,
    get_strategies,
)
from flamby.gpu_utils import use_gpu_idx


def main(args_cli):
    """This function will launch either all single-centric and the FL strategies
    specified in the config file provided.
    This will write the results in a csv file. The FL strategies will use
    hyperparameters defined in the config file or default ones.
    This function behavior can be changed by providing the keywords
    --strategy or --single-centric-baseline which will run only the provided
    strategy or single-centric-baseline alongside parameters input by the CLI.
    One cannot change the parameters of the single-centric baseline which are
    kept fixed.
    Parameters
    ----------
    args_cli : A namespace of hyperparameters providing the ability to overwrite
        the config provided to some extents.

    Returns
    -------
    None
    """
    # Use the same initialization for everyone in order to be fair
    set_seed(args_cli.seed)

    use_gpu = use_gpu_idx(args_cli.GPU, args_cli.cpu_only)
    hyperparameters_names = [
        "learning_rate",
        "server_learning_rate",
        "mu",
        "optimizer-class",
        "deterministic",
    ]
    hyperparameters_changed = [
        e is not None
        for e in [args_cli.learning_rate, args_cli.server_learning_rate, args_cli.mu]
    ] + [args_cli.optimizer_class != "torch.optim.SGD", args_cli.deterministic]
    if (args_cli.strategy is None) and any(hyperparameters_changed):
        hyperparameters_changed = [
            hyperparameters_names[i]
            for i in range(len(hyperparameters_changed))
            if hyperparameters_changed[i]
        ]
        raise ValueError(
            "You cannot change one or several hyperparameters "
            f"({hyperparameters_changed} in your case) in a global fashion for"
            " all strategies, please use the keyword strategy to specify the "
            "strategy you want to affect by writing: "
            "--strategy [FedAvg, FedProx, FedAdam, FedAdagrad, FedYogi, "
            "Cyclic, FedAvgFineTuning]"
            ", otherwise modify the config file directly."
        )
    # Find a way to provide it through hyperparameters
    run_num_updates = [100]

    # ensure that the config provided by the user is ok
    config = check_config(args_cli.config_file_path)

    dataset_name = config["dataset"]

    # get all the dataset specific handles
    (
        FedDataset,
        [
            BATCH_SIZE,
            LR,
            NUM_CLIENTS,
            NUM_EPOCHS_POOLED,
            Baseline,
            BaselineLoss,
            Optimizer,
            get_nb_max_rounds,
            metric,
            collate_fn,
        ],
    ) = get_dataset_args(dataset_name)

    nrounds_list = [get_nb_max_rounds(num_updates) for num_updates in run_num_updates]

    if args_cli.debug:
        nrounds_list = [1 for _ in run_num_updates]
        NUM_EPOCHS_POOLED = 1

    if args_cli.hard_debug:
        nrounds_list = [0 for _ in run_num_updates]
        NUM_EPOCHS_POOLED = 0

    # We can now instantiate the dataset specific model on CPU
    if args_cli.use_ssl_features and dataset_name == "fed_camelyon16":
        global_init = Baseline(768)
    else:
        global_init = Baseline()

    # We parse the hyperparams from the config or from the CLI if strategy is given
    strategy_specific_hp_dicts = get_strategies(
        config, learning_rate=LR, args=vars(args_cli)
    )
    pooled_hyperparameters = {
        "optimizer_class": Optimizer,
        "learning_rate": LR,
        "seed": args_cli.seed,
        "dp_target_epsilon": args_cli.dp_target_epsilon,
        "dp_target_delta": args_cli.dp_target_delta,
        "dp_max_grad_norm": args_cli.dp_max_grad_norm,
    }
    main_columns_names = ["Test", "Method", "Metric", "seed"]

    # We might need to dynamically add additional parameters to the csv columns
    all_strategies_args = []
    # get all hparam names from all the strategies used
    for strategy in strategy_specific_hp_dicts.values():
        all_strategies_args += [
            arg_names
            for arg_names in strategy.keys()
            if arg_names not in all_strategies_args
        ]
    # column names used for the results file
    columns_names = list(set(main_columns_names + all_strategies_args))

    evaluate_func, batch_size_test, compute_ensemble_perf = set_dataset_specific_config(
        dataset_name, compute_ensemble_perf=False
    )

    # We compute the number of local and ensemble performances we should have
    # in the results dataframe
    nb_local_and_ensemble_xps = (NUM_CLIENTS + int(compute_ensemble_perf)) * (
        NUM_CLIENTS + 1
    )

    # We init dataloader for train and test and for local and pooled datasets

    training_dls, test_dls = init_data_loaders(
        dataset=FedDataset,
        pooled=False,
        batch_size=BATCH_SIZE,
        num_workers=args_cli.workers,
        num_clients=NUM_CLIENTS,
        batch_size_test=batch_size_test,
        collate_fn=collate_fn,
    )
    train_pooled, test_pooled = init_data_loaders(
        dataset=FedDataset,
        pooled=True,
        batch_size=BATCH_SIZE,
        num_workers=args_cli.workers,
        batch_size_test=batch_size_test,
        collate_fn=collate_fn,
    )

    # Check if some results are already computed
    results_file = get_results_file(config, path=args_cli.results_file_path)
    if results_file.exists():
        df = pd.read_csv(results_file)
        # Update df if new hyperparameters added
        df = df.reindex(
            df.columns.union(columns_names, sort=False).unique(),
            axis="columns",
            fill_value=None,
        )
    else:
        # initialize data frame with the column_names and no data if no csv was
        # found
        df = pd.DataFrame(columns=columns_names)

    # We compute the experiment plan given the config and user-specific hyperparams

    do_baselines, do_strategy, compute_ensemble_perf = init_xp_plan(
        NUM_CLIENTS,
        args_cli.nlocal,
        args_cli.single_centric_baseline,
        args_cli.strategy,
        compute_ensemble_perf,
    )

    # We can now proceed to the trainings
    # Pooled training
    # We check if we have already the results for pooled
    index_of_interest = df.loc[
        (df["Method"] == "Pooled Training") & (df["seed"] == args_cli.seed)
    ].index

    # There is no use in running the experiment if it is already found
    if (len(index_of_interest) < (NUM_CLIENTS + 1)) and do_baselines["Pooled"]:
        # dealing with edge case that shouldn't happen
        # If some of the rows are there but not all of them we redo the
        # experiments
        if len(index_of_interest) > 0:
            df.drop(index_of_interest, inplace=True)
        m = copy.deepcopy(global_init)
        set_seed(args_cli.seed)
        m = train_single_centric(
            m,
            train_pooled,
            use_gpu,
            "Pooled",
            pooled_hyperparameters["optimizer_class"],
            pooled_hyperparameters["learning_rate"],
            BaselineLoss,
            NUM_EPOCHS_POOLED,
            dp_target_epsilon=pooled_hyperparameters["dp_target_epsilon"],
            dp_target_delta=pooled_hyperparameters["dp_target_delta"],
            dp_max_grad_norm=pooled_hyperparameters["dp_max_grad_norm"],
            seed=args_cli.seed,
        )
        (
            perf_dict,
            pooled_perf_dict,
            _,
            _,
            _,
            _,
        ) = evaluate_model_on_local_and_pooled_tests(
            m, test_dls, test_pooled, metric, evaluate_func
        )
        df = fill_df_with_xp_results(
            df,
            perf_dict,
            pooled_hyperparameters,
            "Pooled Training",
            columns_names,
            results_file,
        )
        df = fill_df_with_xp_results(
            df,
            pooled_perf_dict,
            pooled_hyperparameters,
            "Pooled Training",
            columns_names,
            results_file,
            pooled=True,
        )

    # We check if we have the results for local trainings and possibly ensemble as well
    index_of_interest = df.loc[
        (df["Method"] == "Local 0") & (df["seed"] == args_cli.seed)
    ].index
    for i in range(1, NUM_CLIENTS):
        index_of_interest = index_of_interest.union(
            df.loc[(df["Method"] == f"Local {i}") & (df["seed"] == args_cli.seed)].index
        )
    index_of_interest = index_of_interest.union(
        df.loc[(df["Method"] == "Ensemble") & (df["seed"] == args_cli.seed)].index
    )

    if len(index_of_interest) < nb_local_and_ensemble_xps:
        # The fact that we are here means some local experiments are missing or
        # we need to compute ensemble as well so we need to redo all local experiments
        y_true_dicts = {}
        y_pred_dicts = {}
        pooled_y_true_dicts = {}
        pooled_y_pred_dicts = {}

        for i in range(NUM_CLIENTS):
            index_of_interest = df.loc[
                (df["Method"] == f"Local {i}") & (df["seed"] == args_cli.seed)
            ].index
            # We do the experiments only if results are not found or we need
            # ensemble performances AND this experiment is planned.
            # i.e. we allow to not do anything else if the user specify
            if (
                (len(index_of_interest) < (NUM_CLIENTS + 1)) or compute_ensemble_perf
            ) and do_baselines[f"Local {i}"]:
                if len(index_of_interest) > 0:
                    df.drop(index_of_interest, inplace=True)

                m = copy.deepcopy(global_init)
                method_name = f"Local {i}"
                set_seed(args_cli.seed)
                m = train_single_centric(
                    m,
                    training_dls[i],
                    use_gpu,
                    method_name,
                    pooled_hyperparameters["optimizer_class"],
                    pooled_hyperparameters["learning_rate"],
                    BaselineLoss,
                    NUM_EPOCHS_POOLED,
                    dp_target_epsilon=pooled_hyperparameters["dp_target_epsilon"],
                    dp_target_delta=pooled_hyperparameters["dp_target_delta"],
                    dp_max_grad_norm=pooled_hyperparameters["dp_max_grad_norm"],
                    seed=args_cli.seed,
                )
                (
                    perf_dict,
                    pooled_perf_dict,
                    y_true_dicts[f"Local {i}"],
                    y_pred_dicts[f"Local {i}"],
                    pooled_y_true_dicts[f"Local {i}"],
                    pooled_y_pred_dicts[f"Local {i}"],
                ) = evaluate_model_on_local_and_pooled_tests(
                    m, test_dls, test_pooled, metric, evaluate_func, return_pred=True
                )
                df = fill_df_with_xp_results(
                    df,
                    perf_dict,
                    pooled_hyperparameters,
                    method_name,
                    columns_names,
                    results_file,
                )
                df = fill_df_with_xp_results(
                    df,
                    pooled_perf_dict,
                    pooled_hyperparameters,
                    method_name,
                    columns_names,
                    results_file,
                    pooled=True,
                )

        if compute_ensemble_perf:
            print(
                "Computing ensemble performance, local models need to have been"
                " trained in the same runtime"
            )
            local_ensemble_perf = ensemble_perf_from_predictions(
                y_true_dicts, y_pred_dicts, NUM_CLIENTS, metric
            )
            pooled_ensemble_perf = ensemble_perf_from_predictions(
                pooled_y_true_dicts,
                pooled_y_pred_dicts,
                NUM_CLIENTS,
                metric,
                num_clients_test=1,
            )

            df = fill_df_with_xp_results(
                df,
                local_ensemble_perf,
                pooled_hyperparameters,
                "Ensemble",
                columns_names,
                results_file,
            )
            df = fill_df_with_xp_results(
                df,
                pooled_ensemble_perf,
                pooled_hyperparameters,
                "Ensemble",
                columns_names,
                results_file,
                pooled=True,
            )

    # Strategies
    # Needed for perfect reproducibility otherwise strategies are ordered randomly
    strats_names = list(strategy_specific_hp_dicts.keys())
    strats_names.sort()
    if do_strategy:
        for idx, num_updates in enumerate(run_num_updates):
            for sname in strats_names:
                print("========================")
                print(sname.upper())
                print("========================")
                # Base arguments
                m = copy.deepcopy(global_init)
                bloss = BaselineLoss()
                # We init the strategy parameters to the following default ones
                args = {
                    "training_dataloaders": training_dls,
                    "model": m,
                    "loss": bloss,
                    "optimizer_class": torch.optim.SGD,
                    "learning_rate": LR,
                    "num_updates": num_updates,
                    "nrounds": nrounds_list[idx],
                }
                if sname == "Cyclic":
                    args["rng"] = np.random.default_rng(args_cli.seed)
                # We overwrite defaults with new hyperparameters from config
                strategy_specific_hp_dict = strategy_specific_hp_dicts[sname]
                # Overwriting arguments with strategy specific arguments
                for k, v in strategy_specific_hp_dict.items():
                    args[k] = v
                # We fill the hyperparameters dict for later use in filling
                # the csv by filling missing column with nans
                hyperparameters = {}
                for k in all_strategies_args:
                    if k in args:
                        hyperparameters[k] = args[k]
                    else:
                        hyperparameters[k] = np.nan
                hyperparameters["seed"] = args_cli.seed

                index_of_interest = find_xps_in_df(
                    df, hyperparameters, sname, num_updates
                )

                # An experiment is finished if there are num_clients + 1 rows
                if len(index_of_interest) < (NUM_CLIENTS + 1):
                    # Dealing with edge case that shouldn't happen
                    # If some of the rows are there but not all of them we redo the
                    # experiments
                    if len(index_of_interest) > 0:
                        df.drop(index_of_interest, inplace=True)
                    basename = get_logfile_name_from_strategy(
                        dataset_name, sname, num_updates, args
                    )

                    # We run the FL strategy

                    s = getattr(strats, sname)(
                        **args, log=args_cli.log, log_basename=basename
                    )
                    print("FL strategy", sname, " num_updates ", num_updates)
                    set_seed(args_cli.seed)

                    m = s.run()[0]

                    (
                        perf_dict,
                        pooled_perf_dict,
                        _,
                        _,
                        _,
                        _,
                    ) = evaluate_model_on_local_and_pooled_tests(
                        m, test_dls, test_pooled, metric, evaluate_func
                    )

                    df = fill_df_with_xp_results(
                        df,
                        perf_dict,
                        hyperparameters,
                        sname + str(num_updates),
                        columns_names,
                        results_file,
                    )
                    df = fill_df_with_xp_results(
                        df,
                        pooled_perf_dict,
                        hyperparameters,
                        sname + str(num_updates),
                        columns_names,
                        results_file,
                        pooled=True,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--GPU", type=int, default=0, help="GPU to run the training on (if available)"
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        default=False,
        help="Force computation on CPU.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Do 1 round and 1 epoch to check if the script is working",
    )
    parser.add_argument(
        "--hard-debug",
        action="store_true",
        default=False,
        help="Do 0 round and 0 epoch to check if the script is working",
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="Numbers of workers for the dataloader"
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=None,
        help="Client side learning rate if strategy is given",
    )
    parser.add_argument(
        "--server_learning_rate",
        "-slr",
        type=float,
        default=None,
        help="Server side learning rate if strategy is given",
    )
    parser.add_argument(
        "--mu",
        "-mu",
        type=float,
        default=None,
        help="FedProx mu parameter if strategy is given and that it is FedProx",
    )
    parser.add_argument(
        "--num_fine_tuning_steps",
        "-nft",
        type=int,
        default=None,
        help=(
            "The number of SGD fine-tuning updates to be"
            "performed on the model at the personalization step,"
            "if strategy is given and that it is FedAvgFineTuning"
        ),
    )
    parser.add_argument(
        "--tau",
        "-tau",
        type=float,
        default=None,
        help=(
            "FedOpt tau parameter used only if strategy is "
            "given and that it is a fedopt strategy"
        ),
    )
    parser.add_argument(
        "--beta1",
        "-b1",
        type=float,
        default=None,
        help=(
            "FedOpt beta1 parameter used only if strategy is "
            "given and that it is a fedopt strategy"
        ),
    )
    parser.add_argument(
        "--beta2",
        "-b2",
        type=float,
        default=None,
        help=(
            "FedOpt beta2 parameter used only if strategy is"
            " given and that it is a fedopt strategy"
        ),
    )
    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        default=None,
        help="If this parameter is chosen will only run this specific strategy",
        choices=[
            None,
            "FedAdam",
            "FedYogi",
            "FedAdagrad",
            "Scaffold",
            "FedAvg",
            "Cyclic",
            "FedProx",
            "FedAvgFineTuning",
        ],
    )
    parser.add_argument(
        "--optimizer-class",
        "-opt",
        type=str,
        default="torch.optim.SGD",
        help="The optimizer class to use if strategy is given",
    )
    parser.add_argument(
        "--deterministic",
        "-d",
        action="store_true",
        default=False,
        help="whether or not to use deterministic cycling for the cyclic strategy",
    )
    parser.add_argument(
        "--dp_target_epsilon",
        "-dpe",
        type=float,
        default=None,
        help="the target epsilon for (epsilon, delta)-differential private guarantee",
    )
    parser.add_argument(
        "--dp_target_delta",
        "-dpd",
        type=float,
        default=None,
        help="the target delta for (epsilon, delta)-differential private guarantee",
    )
    parser.add_argument(
        "--dp_max_grad_norm",
        "-mgn",
        type=float,
        default=None,
        help=(
            "the maximum L2 norm of per-sample gradients; "
            "used to enforce differential privacy"
        ),
    )
    parser.add_argument(
        "--log",
        "-l",
        action="store_true",
        default=False,
        help="Whether or not to log the strategies",
    )
    parser.add_argument(
        "--config-file-path",
        "-cfp",
        default="./config.json",
        type=str,
        help="Which config file to use.",
    )
    parser.add_argument(
        "--results-file-path",
        "-rfp",
        default=None,
        type=str,
        help="The path to the created results (overwrite the config path)",
    )
    parser.add_argument(
        "--single-centric-baseline",
        "-scb",
        default=None,
        type=str,
        help=(
            "Whether or not to compute only one single-centric baseline and which one."
        ),
        choices=["Pooled", "Local"],
    )
    parser.add_argument(
        "--nlocal",
        default=0,
        type=int,
        help=(
            "Will only be used if --single-centric-baseline Local, will test"
            "only training on Local {nlocal}."
        ),
    )
    parser.add_argument(
        "--use-ssl-features",
        action="store_true",
        help=(
            "Whether to use the much more performant phikon feature extractor on"
            " Camelyon16, trained with self-supervised learning on histology datasets"
            " from https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v2"
            " instead of imagenet-trained resnet."
        ),
    )
    parser.add_argument("--seed", default=0, type=int, help="Seed")

    args = parser.parse_args()

    main(args)
