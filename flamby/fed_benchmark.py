import argparse
import copy
import os
from xmlrpc.client import Boolean

import pandas as pd
import torch
from torch.utils.data import DataLoader as dl

import flamby.strategies as strats
from flamby.conf import check_config, get_dataset_args, get_results_file, get_strategies
from flamby.utils import evaluate_model_on_tests

# Only 4 lines to change to evaluate different datasets (except for LIDC where the
# evaluation function is custom)
# Still some datasets might require specific augmentation strategies or collate_fn
# functions in the data loading part

NUM_EPOCHS_POOLED = 0  # TODO: remove before merging
# from flamby.datasets.fed_tcga_brca import FedTcgaBrca as FedDataset
# from flamby.datasets.fed_tcga_brca import Optimizer, get_nb_max_rounds, metric


def main(args2):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args2.GPU)
    torch.use_deterministic_algorithms(False)
    use_gpu = args2.GPU and torch.cuda.is_available()
    # ensure that the config is ok
    check_config()
    (
        dataset_name,
        FedDataset,
        [
            BATCH_SIZE,
            LR,
            NUM_CLIENTS,
            NUM_EPOCHS_POOLED,  # change to 0
            Baseline,
            BaselineLoss,
            Optimizer,
            get_nb_max_rounds,
            metric,
        ],
    ) = get_dataset_args(
        [
            "BATCH_SIZE",
            "LR",
            "NUM_CLIENTS",
            "NUM_EPOCHS_POOLED",
            "Baseline",
            "BaselineLoss",
            "Optimizer",
            "get_nb_max_rounds",
            "metric",
        ]
    )
    results_file = get_results_file()

    # One might need to iterate on the hyperparameters to some extent if performances
    # are seriously degraded with default ones
    strategy_specific_hp_dicts = get_strategies(learning_rate=LR)

    columns_names = ["Test", "Method", "Metric"]
    # We need to add strategy hyperparameters columns to the benchmark
    hp_additional_args = []

    for strategy in strategy_specific_hp_dicts.values():
        hp_additional_args += [
            arg_names
            for arg_names in strategy.keys()
            if arg_names not in hp_additional_args
        ]
    columns_names += hp_additional_args
    base_dict = {col_name: None for col_name in columns_names}

    # We use the same initialization for everyone in order to be fair
    torch.manual_seed(0)
    global_init = Baseline()

    # We instantiate all train and test dataloaders required including pooled ones
    training_dls, test_dls = init_data_loaders(
        dataset=FedDataset,
        pooled=False,
        batch_size=BATCH_SIZE,
        num_workers=args2.workers,
        num_clients=NUM_CLIENTS,
    )
    train_pooled, test_pooled = init_data_loaders(
        dataset=FedDataset,
        pooled=True,
        batch_size=BATCH_SIZE,
        num_workers=args2.workers,
    )

    # We check if some results are already computed
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        # If we added additional hyperparameters we update the df
        for col_name in columns_names:
            if col_name not in df.columns:
                df[col_name] = None
        perf_lines_dicts = df.to_dict("records")
    else:
        df = pd.DataFrame({k: [v] for k, v in base_dict.items()})
        perf_lines_dicts = []

    # Single client baseline computation
    # We use the same set of parameters as found in the corresponding
    # flamby/datasets/fed_mydataset/benchmark.py

    # Pooled Baseline
    # Throughout the experiments we only launch training if we do not have the results
    # yet. Note that pooled and local baselines do not use hyperparameters.

    index_of_interest = df.loc[df["Method"] == "Pooled Training"].index
    # an experiment is finished if there are num_clients + 1 rows
    if len(index_of_interest) < (NUM_CLIENTS + 1):
        # dealing with edge case that shouldn't happen
        # If some of the rows are there but not all of them we redo the experiments
        if len(index_of_interest) > 0:
            df.drop(index_of_interest, inplace=True)
            perf_lines_dicts = df.to_dict("records")
        m = copy.deepcopy(global_init)
        if use_gpu:
            m.cuda()
        bloss = BaselineLoss()
        opt = Optimizer(m.parameters(), lr=LR)
        print("Pooled")
        for _ in range(NUM_EPOCHS_POOLED):
            for X, y in train_pooled:  # CUDA if GPUT X = X.cuda() (same for y)
                if use_gpu:
                    X = X.cuda()
                    y = y.cuda()
                opt.zero_grad()
                y_pred = m(X)
                loss = bloss(y_pred, y)
                loss.backward()
                opt.step()

        perf_dict = evaluate_model_on_tests(m, test_dls, metric, use_gpu=use_gpu)
        pooled_perf_dict = evaluate_model_on_tests(
            m, [test_pooled], metric, use_gpu=use_gpu
        )
        for k, v in perf_dict.items():
            # Make sure there is no weird inplace stuff
            current_dict = copy.deepcopy(base_dict)
            current_dict["Test"] = k
            current_dict["Metric"] = v
            current_dict["Method"] = "Pooled Training"
            current_dict["learning_rate"] = str(LR)
            current_dict["optimizer_class"] = Optimizer
            perf_lines_dicts.append(current_dict)
        current_dict = copy.deepcopy(base_dict)
        current_dict["Test"] = "Pooled Test"
        current_dict["Metric"] = pooled_perf_dict["client_test_0"]
        current_dict["Method"] = "Pooled Training"
        current_dict["learning_rate"] = str(LR)
        current_dict["optimizer_class"] = Optimizer
        perf_lines_dicts.append(current_dict)
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

    if len(index_of_interest) < (NUM_CLIENTS + 1) ** 2:
        # Dealing with edge case that shouldn't happen.
        # If some of the rows are there but not all of them we redo the experiments.
        if len(index_of_interest) > 0:
            df.drop(index_of_interest, inplace=True)
            perf_lines_dicts = df.to_dict("records")

        for i in range(NUM_CLIENTS):
            m = copy.deepcopy(global_init)
            bloss = BaselineLoss()
            print(LR)
            opt = Optimizer(m.parameters(), lr=LR)
            print("Local " + str(i))
            for e in range(NUM_EPOCHS_POOLED):
                for X, y in training_dls[i]:
                    opt.zero_grad()
                    y_pred = m(X)
                    loss = bloss(y_pred, y)
                    loss.backward()
                    opt.step()

            (
                perf_dict,
                y_true_dicts[f"Local {i}"],
                y_pred_dicts[f"Local {i}"],
            ) = evaluate_model_on_tests(m, test_dls, metric, return_pred=True)
            (
                pooled_perf_dict,
                pooled_y_true_dicts[f"Local {i}"],
                pooled_y_pred_dicts[f"Local {i}"],
            ) = evaluate_model_on_tests(m, [test_pooled], metric, return_pred=True)

            for k, v in perf_dict.items():
                # Make sure there is no weird inplace stuff
                current_dict = copy.deepcopy(base_dict)
                current_dict["Test"] = k
                current_dict["Metric"] = v
                current_dict["Method"] = f"Local {i}"
                current_dict["learning_rate"] = str(LR)
                current_dict["optimizer_class"] = Optimizer
                perf_lines_dicts.append(current_dict)
            current_dict = copy.deepcopy(base_dict)
            current_dict["Test"] = "Pooled Test"
            current_dict["Metric"] = pooled_perf_dict["client_test_0"]
            current_dict["Method"] = f"Local {i}"
            current_dict["learning_rate"] = str(LR)
            current_dict["optimizer_class"] = Optimizer
            perf_lines_dicts.append(current_dict)

        for testset in range(NUM_CLIENTS):
            for model in range(1, NUM_CLIENTS):
                assert (
                    y_true_dicts[f"Local {0}"][f"client_test_{testset}"]
                    == y_true_dicts[f"Local {model}"][f"client_test_{testset}"]
                ).all(), (
                    "Models in the ensemmble do not make predictions in the same x order"
                )
            ensemble_true = y_true_dicts["Local 0"][f"client_test_{testset}"]
            ensemble_pred = y_pred_dicts["Local 0"][f"client_test_{testset}"]
            for model in range(1, NUM_CLIENTS):
                ensemble_pred += y_pred_dicts[f"Local {model}"][f"client_test_{testset}"]
            ensemble_pred /= NUM_CLIENTS

            current_dict = copy.deepcopy(base_dict)
            current_dict["Test"] = f"client_test_{testset}"
            current_dict["Metric"] = metric(ensemble_true, ensemble_pred)
            current_dict["Method"] = "Ensemble"
            current_dict["learning_rate"] = str(LR)
            current_dict["optimizer_class"] = Optimizer
            perf_lines_dicts.append(current_dict)

        for model in range(1, NUM_CLIENTS):
            assert (
                pooled_y_true_dicts["Local 0"]["client_test_0"]
                == pooled_y_true_dicts[f"Local {model}"]["client_test_0"]
            ).all(), (
                "Models in the ensemmble do not make predictions in the same x order"
            )
        pooled_ensemble_true = pooled_y_true_dicts["Local 0"]["client_test_0"]
        pooled_ensemble_pred = pooled_y_pred_dicts["Local 0"]["client_test_0"]
        for model in range(1, NUM_CLIENTS):
            pooled_ensemble_pred += pooled_y_pred_dicts[f"Local {model}"][
                "client_test_0"
            ]
        pooled_ensemble_pred /= NUM_CLIENTS

        current_dict = copy.deepcopy(base_dict)
        current_dict["Test"] = "Pooled Test"
        current_dict["Metric"] = metric(pooled_ensemble_true, pooled_ensemble_pred)
        current_dict["Method"] = "Ensemble"
        current_dict["learning_rate"] = str(LR)
        current_dict["optimizer_class"] = Optimizer
        perf_lines_dicts.append(current_dict)

        # We update csv and save it when the results are there
        df = pd.DataFrame.from_dict(perf_lines_dicts)
        df.to_csv(results_file, index=False)

    # Strategies
    for num_updates in [1, 10, 100, 500]:
        for sname in strategy_specific_hp_dicts.keys():
            # Base arguments
            m = copy.deepcopy(global_init)
            bloss = BaselineLoss()
            args = {
                "training_dataloaders": training_dls,
                "model": m,
                "loss": bloss,
                "optimizer_class": Optimizer,
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
                    hyperparameters[k] = str(args[k])
                else:
                    hyperparameters[k] = "nan"
            index_of_interest = df.loc[
                (df["Method"] == (sname + str(num_updates)))
                & (
                    df[list(hyperparameters)].astype(str) == pd.Series(hyperparameters)
                ).all(axis=1)
            ].index
            # An experiment is finished if there are num_clients + 1 rows
            if len(index_of_interest) < (NUM_CLIENTS + 1):
                # Dealing with edge case that shouldn't happen
                # If some of the rows are there but not all of them we redo the
                # experiments
                if len(index_of_interest) > 0:
                    df.drop(index_of_interest, inplace=True)
                    perf_lines_dicts = df.to_dict("records")
                # We run the FL strategy
                s = getattr(strats, sname)(**args)
                print("FL strategy", sname, " num_updates ", num_updates)
                m = s.run()[0]

                perf_dict = evaluate_model_on_tests(m, test_dls, metric)
                pooled_perf_dict = evaluate_model_on_tests(m, [test_pooled], metric)
                for k, v in perf_dict.items():
                    # Make sure there is no weird inplace stuff
                    current_dict = copy.deepcopy(base_dict)
                    current_dict["Test"] = k
                    current_dict["Metric"] = v
                    current_dict["Method"] = sname + str(num_updates)
                    # We add the hyperparameters used
                    for k2, v2 in hyperparameters.items():
                        if k2 not in ["Test", "Metric", "Method"]:
                            current_dict[k2] = v2
                    perf_lines_dicts.append(current_dict)
                current_dict = copy.deepcopy(base_dict)
                current_dict["Test"] = "Pooled Test"
                current_dict["Metric"] = pooled_perf_dict["client_test_0"]
                current_dict["Method"] = sname + str(num_updates)
                # We add the hyperparamters used
                for k2, v2 in hyperparameters.items():
                    if k2 not in ["Test", "Metric", "Method"]:
                        current_dict[k2] = v2
                perf_lines_dicts.append(current_dict)
                # We update csv and save it when the results are there
                df = pd.DataFrame.from_dict(perf_lines_dicts)
                df.to_csv(results_file, index=False)


def init_data_loaders(
    dataset, pooled=False, batch_size=1, num_workers=1, num_clients=None
):
    """
    Initializes the data loaders for the training and test datasets.
    """
    if (not pooled) and num_clients is None:
        raise ValueError("num_clients must be specified for the non-pooled data")

    if not pooled:
        training_dls = [
            dl(
                dataset(center=i, train=True, pooled=False),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
            for i in range(num_clients)
        ]
        test_dls = [
            dl(
                dataset(center=i, train=False, pooled=False),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
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
        )
        test_pooled = dl(
            dataset(train=False, pooled=True),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return train_pooled, test_pooled


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--GPU",
        type=Boolean,
        default=False,
        help="GPU to run the training on (if available)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Numbers of workers for the dataloader",
    )
    args2 = parser.parse_args()

    main(args2)
