import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader as dl
from tqdm import tqdm

import flamby.strategies as strats
from flamby.conf import check_config, get_dataset_args, get_results_file, get_strategies
from flamby.utils import evaluate_model_on_tests


# Only 4 lines to change to evaluate different datasets (except for LIDC where the
# evaluation function is custom)
# Still some datasets might require specific augmentation strategies or collate_fn
# functions in the data loading part
def main(args_cli):
    n_gpus = torch.cuda.device_count()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_cli.GPU)
    # torch.use_deterministic_algorithms(False)
    use_gpu = (
        args_cli.GPU in [str(i) for i in range(n_gpus)]
    ) and torch.cuda.is_available()
    run_num_updates = [100, 500]

    # ensure that the config is ok
    config = check_config(args_cli.config_file_path)

    params_list = [
        "BATCH_SIZE",
        "LR",
        "NUM_CLIENTS",
        "NUM_EPOCHS_POOLED",
        "Baseline",
        "BaselineLoss",
        "Optimizer",
        "get_nb_max_rounds",
        "metric",
        "collate_fn",
    ]
    # get all the dataset args
    (
        dataset_name,
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
    ) = get_dataset_args(config, params_list)
    results_file = get_results_file(config, path=args_cli.results_file_path)

    # One might need to iterate on the hyperparameters to some extent if performances
    # are seriously degraded with default ones
    strategy_specific_hp_dicts = get_strategies(
        config, learning_rate=LR, args=vars(args_cli)
    )

    init_hp_additional_args = ["Test", "Method", "Metric"]
    # We need to add strategy hyperparameters columns to the benchmark
    hp_additional_args = []

    # get all hparam names from all the strategies used
    for strategy in strategy_specific_hp_dicts.values():
        hp_additional_args += [
            arg_names
            for arg_names in strategy.keys()
            if arg_names not in hp_additional_args
        ]
    # column names used for the results file
    columns_names = init_hp_additional_args + hp_additional_args

    # Use the same initialization for everyone in order to be fair
    torch.manual_seed(0)
    global_init = Baseline()
    # Instantiate all train and test dataloaders required including pooled ones
    if dataset_name == "fed_lidc_idri":
        batch_size_test = 1
        from flamby.datasets.fed_lidc_idri import evaluate_dice_on_tests_by_chunks

        def evaluate_func(m, test_dls, metric, use_gpu=use_gpu, return_pred=False):
            dice_dict = evaluate_dice_on_tests_by_chunks(m, test_dls, use_gpu)
            # dice_dict = {f"client_test_{i}": 0.5 for i in range(NUM_CLIENTS)}
            if return_pred:
                return dice_dict, None, None
            return dice_dict

        compute_ensemble_perf = False
    elif dataset_name == "fed_kits19":
        from flamby.datasets.fed_kits19 import evaluate_dice_on_tests

        batch_size_test = 2

        def evaluate_func(m, test_dls, metric, use_gpu=use_gpu, return_pred=False):
            dice_dict = evaluate_dice_on_tests(m, test_dls, metric, use_gpu)
            # dice_dict = {f"client_test_{i}": 0.5 for i in range(NUM_CLIENTS)}
            if return_pred:
                return dice_dict, None, None
            return dice_dict

        compute_ensemble_perf = False

    else:
        batch_size_test = None
        evaluate_func = evaluate_model_on_tests
        compute_ensemble_perf = False

    nb_local_and_ensemble_xps = (NUM_CLIENTS + int(compute_ensemble_perf)) * (
        NUM_CLIENTS + 1
    )

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
    if results_file.exists():
        df = pd.read_csv(results_file)
        # Update df if new hyperparameters added
        df = df.reindex(
            df.columns.union(columns_names, sort=False), axis="columns", fill_value=None
        )
        perf_lines_dicts = df.to_dict("records")
    else:
        # initialize data frame with the column_names and no data
        df = pd.DataFrame(columns=columns_names)
        perf_lines_dicts = []

    do_strategy = True
    do_baseline = {"Pooled": True}
    for i in range(NUM_CLIENTS):
        do_baseline[f"Local {i}"] = True
    # Single client baseline computation
    if args_cli.single_centric_baseline is not None:
        do_baseline = {"Pooled": False}
        for i in range(NUM_CLIENTS):
            do_baseline[f"Local {i}"] = False
        if args_cli.single_centric_baseline == "Pooled":
            do_baseline[args_cli.single_centric_baseline] = True
        elif args_cli.single_centric_baseline == "Local":
            assert args_cli.nlocal in range(
                NUM_CLIENTS
            ), "The client you chose does not exist"
            do_baseline[
                args_cli.single_centric_baseline + " " + str(args_cli.nlocal)
            ] = True
        # If we do a single-centric baseline we don't do the strategies
        do_strategy = False

    # if we give a strategy we compute only the strategy and not the baselines
    if args_cli.strategy is not None:
        for k, _ in do_baseline.items():
            do_baseline[k] = False

    do_all_local = all([do_baseline[f"Local {i}"] for i in range(NUM_CLIENTS)])
    if compute_ensemble_perf and not (do_all_local):
        raise ValueError(
            "Cannot compute ensemble performance if training on only one local"
        )

    # We use the same set of parameters as found in the corresponding
    # flamby/datasets/fed_mydataset/benchmark.py
    # Pooled Baseline
    # Throughout the experiments we only launch training if we do not have the results
    # yet. Note that pooled and local baselines do not use hyperparameters.
    index_of_interest = df.loc[df["Method"] == "Pooled Training"].index
    # an experiment is finished if there are num_clients + 1 rows
    if (len(index_of_interest) < (NUM_CLIENTS + 1)) and do_baseline["Pooled"]:
        # dealing with edge case that shouldn't happen
        # If some of the rows are there but not all of them we redo the experiments
        if len(index_of_interest) > 0:
            df.drop(index_of_interest, inplace=True)
            perf_lines_dicts = df.to_dict("records")
        model = copy.deepcopy(global_init)
        if use_gpu:
            model.cuda()
        bloss = BaselineLoss()
        opt = Optimizer(model.parameters(), lr=LR)
        print("Pooled")
        for _ in tqdm(range(NUM_EPOCHS_POOLED)):
            for X, y in train_pooled:
                if use_gpu:
                    # use GPU if requested and available
                    X = X.cuda()
                    y = y.cuda()
                opt.zero_grad()
                y_pred = model(X)
                loss = bloss(y_pred, y)
                loss.backward()
                opt.step()

        perf_dict = evaluate_func(model, test_dls, metric, use_gpu=use_gpu)
        pooled_perf_dict = evaluate_func(model, [test_pooled], metric, use_gpu=use_gpu)
        print("Per-center performance:")
        print(perf_dict)
        print("Performance on pooled test set:")
        print(pooled_perf_dict)
        method = "Pooled Training"
        for k, v in perf_dict.items():
            perf_lines_dicts.append(
                prepare_dict(
                    keys=columns_names,
                    Test=k,
                    Metric=v,
                    Method=method,
                    learning_rate=str(LR),
                    optimizer_class=Optimizer,
                )
            )

        perf_lines_dicts.append(
            prepare_dict(
                keys=columns_names,
                Test="Pooled Test",
                Metric=pooled_perf_dict["client_test_0"],
                Method=method,
                learning_rate=str(LR),
                optimizer_class=Optimizer,
            )
        )
        # We update csv and save it when the results are there
        df = pd.DataFrame.from_dict(perf_lines_dicts)
        df.to_csv(results_file, index=False)

    # Local baselines and ensemble
    y_true_dicts = {}
    y_pred_dicts = {}
    pooled_y_true_dicts = {}
    pooled_y_pred_dicts = {}

    # We only launch training if it's not finished already.
    index_of_interest = df.loc[df["Method"] == "Local 0"].index
    for i in range(1, NUM_CLIENTS):
        index_of_interest = index_of_interest.union(
            df.loc[df["Method"] == f"Local {i}"].index
        )
    index_of_interest = index_of_interest.union(df.loc[df["Method"] == "Ensemble"].index)
    # This experiment is finished if there are num_clients + 1 rows in each local
    # training and the ensemble training

    if len(index_of_interest) < nb_local_and_ensemble_xps:
        # Dealing with edge case that shouldn't happen.
        # If some of the rows are there but not all of them we redo the experiments.
        for i in range(NUM_CLIENTS):
            m = copy.deepcopy(global_init)
            if use_gpu:
                m = m.cuda()
            bloss = BaselineLoss()
            opt = Optimizer(m.parameters(), lr=LR)
            index_of_interest = df.loc[df["Method"] == f"Local {i}"].index

            if (
                (len(index_of_interest) < (NUM_CLIENTS + 1)) or compute_ensemble_perf
            ) and do_baseline[f"Local {i}"]:
                if len(index_of_interest) > 0:
                    df.drop(index_of_interest, inplace=True)
                    perf_lines_dicts = df.to_dict("records")
                print("Local " + str(i))
                for e in tqdm(range(NUM_EPOCHS_POOLED)):
                    for X, y in training_dls[i]:
                        if use_gpu:
                            X = X.cuda()
                            y = y.cuda()
                        opt.zero_grad()
                        y_pred = m(X)
                        loss = bloss(y_pred, y)
                        loss.backward()
                        opt.step()

                (
                    perf_dict,
                    y_true_dicts[f"Local {i}"],
                    y_pred_dicts[f"Local {i}"],
                ) = evaluate_func(m, test_dls, metric, return_pred=True)
                (
                    pooled_perf_dict,
                    pooled_y_true_dicts[f"Local {i}"],
                    pooled_y_pred_dicts[f"Local {i}"],
                ) = evaluate_func(m, [test_pooled], metric, return_pred=True)
                print("Per-center performance:")
                print(perf_dict)
                print("Performance on pooled test set:")
                print(pooled_perf_dict)
                for k, v in perf_dict.items():
                    # Make sure there is no weird inplace stuff
                    perf_lines_dicts.append(
                        prepare_dict(
                            keys=columns_names,
                            Test=k,
                            Metric=v,
                            Method=f"Local {i}",
                            learning_rate=str(LR),
                            optimizer_class=Optimizer,
                        )
                    )
                perf_lines_dicts.append(
                    prepare_dict(
                        keys=columns_names,
                        Test="Pooled Test",
                        Metric=pooled_perf_dict["client_test_0"],
                        Method=f"Local {i}",
                        learning_rate=str(LR),
                        optimizer_class=Optimizer,
                    )
                )
                # We update csv and save it when the results are there
                df = pd.DataFrame.from_dict(perf_lines_dicts)
                df.to_csv(results_file, index=False)

        if compute_ensemble_perf:
            print("Computing ensemble performance")
            for testset in range(NUM_CLIENTS):
                for model in range(1, NUM_CLIENTS):
                    assert (
                        y_true_dicts[f"Local {0}"][f"client_test_{testset}"]
                        == y_true_dicts[f"Local {model}"][f"client_test_{testset}"]
                    ).all(), "Models in the ensemble have different ground truths"
                ensemble_true = y_true_dicts["Local 0"][f"client_test_{testset}"]
                ensemble_pred = y_pred_dicts["Local 0"][f"client_test_{testset}"]
                for model in range(1, NUM_CLIENTS):
                    ensemble_pred += y_pred_dicts[f"Local {model}"][
                        f"client_test_{testset}"
                    ]
                ensemble_pred /= NUM_CLIENTS

                perf_lines_dicts.append(
                    prepare_dict(
                        keys=columns_names,
                        Test=f"client_test_{testset}",
                        Metric=metric(ensemble_true, ensemble_pred),
                        Method="Ensemble",
                        learning_rate=str(LR),
                        optimizer_class=Optimizer,
                    )
                )
                # We update csv and save it when the results are there
                df = pd.DataFrame.from_dict(perf_lines_dicts)
                df.to_csv(results_file, index=False)
            # Computing ensemble performance in the pooled case
            for model in range(1, NUM_CLIENTS):
                assert (
                    pooled_y_true_dicts["Local 0"]["client_test_0"]
                    == pooled_y_true_dicts[f"Local {model}"]["client_test_0"]
                ).all(), (
                    "Models in the ensemble do not make predictions in the same x order"
                )
            pooled_ensemble_true = pooled_y_true_dicts["Local 0"]["client_test_0"]
            pooled_ensemble_pred = pooled_y_pred_dicts["Local 0"]["client_test_0"]
            for model in range(1, NUM_CLIENTS):
                pooled_ensemble_pred += pooled_y_pred_dicts[f"Local {model}"][
                    "client_test_0"
                ]
            pooled_ensemble_pred /= NUM_CLIENTS

            perf_lines_dicts.append(
                prepare_dict(
                    keys=columns_names,
                    Test="Pooled Test",
                    Metric=metric(pooled_ensemble_true, pooled_ensemble_pred),
                    Method="Ensemble",
                    learning_rate=str(LR),
                    optimizer_class=Optimizer,
                )
            )

            # We update csv and save it when the results are there
            df = pd.DataFrame.from_dict(perf_lines_dicts)
            df.to_csv(results_file, index=False)

    # Strategies
    if do_strategy:
        for num_updates in run_num_updates:
            for sname in strategy_specific_hp_dicts.keys():
                # Base arguments
                m = copy.deepcopy(global_init)
                bloss = BaselineLoss()
                args = {
                    "training_dataloaders": training_dls,
                    "model": m,
                    "loss": bloss,
                    "optimizer_class": torch.optim.SGD,
                    "learning_rate": LR,
                    "num_updates": num_updates,
                    "nrounds": get_nb_max_rounds(num_updates),
                }
                strategy_specific_hp_dict = strategy_specific_hp_dicts[sname]
                # Overwriting arguments with strategy specific arguments
                for k, v in strategy_specific_hp_dict.items():
                    args[k] = v
                # We only launch training if it's not finished already. Maybe FL
                # hyperparameters need to be tuned.
                hyperparameters = {}
                for k in hp_additional_args:  # columns_names:
                    if k in args:
                        hyperparameters[k] = args[k]
                    else:
                        hyperparameters[k] = np.nan
                # This is very ugly but this is the only way I found to accomodate float
                # and objects equality in a robust fashion
                found_xps = df[list(hyperparameters)]
                found_xps_numerical = found_xps.select_dtypes(exclude=[object])
                if 'deterministic_cycle' in found_xps_numerical.columns:
                    found_xps_numerical['deterministic_cycle'] = found_xps_numerical['deterministic_cycle'].fillna(0.0).astype(float)
                col_numericals = found_xps_numerical.columns
                col_objects = [c for c in found_xps.columns if not (c in col_numericals)]

                if len(col_numericals) > 0:
                    bool_numerical = np.all(
                        np.isclose(
                            found_xps_numerical,
                            pd.Series(
                                {
                                    k: float(hyperparameters[k])
                                    for k in list(hyperparameters.keys())
                                    if k in col_numericals
                                }
                            ),
                            equal_nan=True,
                        ),
                        axis=1,
                    )
                else:
                    bool_numerical = np.ones((len(df.index), 1)).astype("bool")

                if len(col_objects):
                    bool_objects = found_xps[col_objects].astype(str) == pd.Series(
                        {
                            k: str(hyperparameters[k])
                            for k in list(hyperparameters.keys())
                            if k in col_objects
                        }
                    )
                else:
                    bool_objects = np.ones((len(df.index), 1)).astype("bool")

                bool_method = df["Method"] == (sname + str(num_updates))
                index_of_interest_1 = df.loc[
                    pd.DataFrame(bool_numerical).all(axis=1)
                ].index
                index_of_interest_2 = df.loc[
                    pd.DataFrame(bool_objects).all(axis=1)
                ].index
                index_of_interest_3 = df.loc[pd.DataFrame(bool_method).all(axis=1)].index
                index_of_interest = index_of_interest_1.intersection(
                    index_of_interest_2
                ).intersection(index_of_interest_3)

                # non-robust version
                # index_of_interest = df.loc[
                #   (df["Method"] == (sname + str(num_updates)))
                #   & (
                #       df[list(hyperparameters)] == pd.Series(hyperparameters)
                #   ).all(axis=1)
                # ].index
                # An experiment is finished if there are num_clients + 1 rows
                if len(index_of_interest) < (NUM_CLIENTS + 1):
                    # Dealing with edge case that shouldn't happen
                    # If some of the rows are there but not all of them we redo the
                    # experiments
                    if len(index_of_interest) > 0:
                        df.drop(index_of_interest, inplace=True)
                        perf_lines_dicts = df.to_dict("records")
                    basename = dataset_name + "-" + sname + f"-num-updates{num_updates}"
                    for k, v in args.items():
                        if k in ["learning_rate", "server_learning_rate"]:
                            basename += (
                                "-" + "".join([e[0] for e in str(k).split("_")]) + str(v)
                            )
                        if k in ["mu", "deterministic_cycle"]:
                            basename += "-" + str(k) + str(v)

                    # We run the FL strategy
                    s = getattr(strats, sname)(
                        **args, log=args_cli.log, log_basename=basename
                    )
                    print("FL strategy", sname, " num_updates ", num_updates)
                    m = s.run()[0]

                    perf_dict = evaluate_func(m, test_dls, metric)
                    pooled_perf_dict = evaluate_func(m, [test_pooled], metric)
                    print("Per-center performance:")
                    print(perf_dict)
                    print("Performance on pooled test set:")
                    print(pooled_perf_dict)
                    hyperparams_save = {
                        k: v
                        for k, v in hyperparameters.items()
                        if k not in init_hp_additional_args
                    }
                    for k, v in perf_dict.items():
                        perf_lines_dicts.append(
                            prepare_dict(
                                keys=columns_names,
                                allow_new=True,
                                Test=k,
                                Metric=v,
                                Method=sname + str(num_updates),
                                # We add the hyperparameters used
                                **hyperparams_save,
                            )
                        )
                    perf_lines_dicts.append(
                        prepare_dict(
                            keys=columns_names,
                            allow_new=True,
                            Test="Pooled Test",
                            Metric=pooled_perf_dict["client_test_0"],
                            Method=sname + str(num_updates),
                            # We add the hyperparamters used
                            **hyperparams_save,
                        )
                    )

                    # We update csv and save it when the results are there
                    df = pd.DataFrame.from_dict(perf_lines_dicts)
                    df.to_csv(results_file, index=False)


def init_data_loaders(
    dataset,
    pooled=False,
    batch_size=1,
    num_workers=1,
    num_clients=None,
    batch_size_test=None,
    collate_fn=None,
):
    """
    Initializes the data loaders for the training and test datasets.
    """
    if (not pooled) and num_clients is None:
        raise ValueError("num_clients must be specified for the non-pooled data")
    batch_size_test = batch_size if batch_size_test is None else batch_size_test
    if not pooled:
        training_dls = [
            dl(
                dataset(center=i, train=True, pooled=False),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )
            for i in range(num_clients)
        ]
        test_dls = [
            dl(
                dataset(center=i, train=False, pooled=False),
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )
            for i in range(num_clients)
        ]
        return training_dls, test_dls
    else:
        train_pooled = dl(
            dataset(train=True, pooled=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        test_pooled = dl(
            dataset(train=False, pooled=True),
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        return train_pooled, test_pooled


def prepare_dict(keys, allow_new=False, **kwargs):
    """
    Prepares the dictionary with the given keys and fills them with the kwargs.
    If allow_new is set to False (default)
    Kwargs must be one of the keys. If
    kwarg is not given for a key the value of that key will be None
    """
    if not allow_new:
        # ensure all the kwargs are in the columns
        assert sum([not (key in keys) for key in kwargs.keys()]) == 0, (
            "Some of the keys given were not found in the existsing columns;"
            f"keys: {kwargs.keys()}, columns: {keys}"
        )

    # create the dictionary from the given keys and fill when appropriate with the kwargs
    return {**dict.fromkeys(keys), **kwargs}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--GPU",
        type=str,
        default="0",
        help="GPU to run the training on (if available)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Numbers of workers for the dataloader",
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
        "--log",
        "-l",
        action="store_true",
        default=False,
        help="Whether or not to log the strategies",
    )
    parser.add_argument(
        "--config-file-path",
        "-cfgp",
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
        help="Whether or not to compute only one single-centric baseline and which one.",
        choices=["Pooled", "Local"],
    )
    parser.add_argument(
        "--nlocal",
        default=0,
        type=int,
        help="Will only be used if --single-centric-baseline Local, will test"
        "only training on Local {nlocal}.",
    )

    args = parser.parse_args()

    main(args)
