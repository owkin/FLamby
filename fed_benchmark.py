import argparse
import copy
import os
import ipdb
import pandas as pd
import torch
from torch.utils.data import DataLoader as dl

import flamby.strategies as strats

# Only 3 lines to change to evaluate different datasets (except for LIDC where the
# evaluation function is custom)
# Still some datasets might require specific augmentation strategies or collate_fn
# functions in the data loading part
from flamby.datasets.fed_tcga_brca import (
    BATCH_SIZE,
    LR,
    NUM_CLIENTS,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
)
from flamby.datasets.fed_tcga_brca import FedTcgaBrca as FedDataset
from flamby.datasets.fed_tcga_brca import Optimizer, get_nb_max_rounds, metric
from flamby.utils import evaluate_model_on_tests


def main(args2):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args2.GPU)
    torch.use_deterministic_algorithms(False)

    NAME_RESULTS_FILE = "results_benchmark.csv"

    #strategy_names = ["FedAvg", "FedProx", "FedAdagrad", "FedAdam", "FedYogi", "Cyclic", "Scaffold"]
    strategy_names = ["FedAvg", "FedProx"]

    # One might need to iterate on the hyperparameters to some extents if performances
    # are seriously degraded with default ones
    # We can add parameters or change them on the go, in the future an argparse could
    # be used to make the process easier
    strategy_specific_hp_dicts = {}
    strategy_specific_hp_dicts["FedAvg"] = {}
    strategy_specific_hp_dicts["FedProx"] = {"learning_rate": LR / 10.0, "mu": 1e-1}
    strategy_specific_hp_dicts["Cyclic"] = {"learning_rate": LR / 100.0}
    strategy_specific_hp_dicts["FedAdam"] = {
        "learning_rate": LR / 10.0,
        "tau": 1e-8,
        "server_learning_rate": 1e-1,
        "beta1": 0.9,
        "beta2": 0.999,
        "optimizer_class": torch.optim.SGD,
    }
    strategy_specific_hp_dicts["FedYogi"] = {
        "learning_rate": LR / 10.0,
        "tau": 1e-8,
        "server_learning_rate": 1e-1,
        "beta1": 0.9,
        "beta2": 0.999,
        "optimizer_class": torch.optim.SGD,
    }
    strategy_specific_hp_dicts["FedAdagrad"] = {
        "learning_rate": LR / 10.0,
        "tau": 1e-8,
        "server_learning_rate": 1e-1,
        "beta1": 0.9,
        "beta2": 0.999,
        "optimizer_class": torch.optim.SGD,
    }
    strategy_specific_hp_dicts["Scaffold"] = {
        "server_learning_rate": 1.0,
        "update_rule": "II",
    }

    columns_names = ["Test", "Method", "Metric"]
    # We need to add strategy hyperparameters columns to the benchmark
    hp_additional_args = []
    for _, v in strategy_specific_hp_dicts.items():
        for name, _ in v.items():
            hp_additional_args.append(name)
    columns_names += hp_additional_args
    #ipdb.set_trace()

    # We instantiate all train and test dataloaders required including pooled ones

    # FIX the number of workers ! parallelism issues when > 0
    training_dls = [
        dl(
            FedDataset(center=i, train=True, pooled=False),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=args2.workers,
        )
        for i in range(NUM_CLIENTS)
    ]
    test_dls = [
        dl(
            FedDataset(center=i, train=False, pooled=False),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=args2.workers,
        )
        for i in range(NUM_CLIENTS)
    ]

    # We use the same initialization for everyone in order to be fair
    torch.manual_seed(42)
    global_init = Baseline()

    train_pooled = dl(
        FedDataset(train=True, pooled=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args2.workers,
    )
    test_pooled = dl(
        FedDataset(train=True, pooled=True),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args2.workers,
    )

    # We check if some results are already computed
    if os.path.exists(NAME_RESULTS_FILE):
        df = pd.read_csv(NAME_RESULTS_FILE)
        #ipdb.set_trace()
        # If we added additional hyperparameters we update the df
        for col_name in columns_names:
            if col_name not in df.columns:
                df[col_name] = None
        perf_lines_dicts = df.to_dict("records")
        base_dict = {col_name: None for col_name in columns_names}
    else:
        base_dict = {col_name: None for col_name in columns_names}
        df = pd.DataFrame({k: [v] for k, v in base_dict.items()})
        perf_lines_dicts = []

    # Single client baseline computation
    # We use the same set of parameters as found in the corresponding
    # flamby/datasets/fed_mydataset/benchmark.py

    # Pooled Baseline
    # Throughout the experiments we only launch training if we do not have the results
    # already. Note that pooled and local baselines do not use hyperparameters.

    index_of_interest = df.loc[df["Method"] == "Pooled Training"].index
    # an experiment is finished if there are num_clients + 1 rows
    if len(index_of_interest) < (NUM_CLIENTS + 1):
        # dealing with edge case that shouldn't happen
        # If some of the rows are there but not all of them we redo the experiments
        if len(index_of_interest) > 0:
            df.drop(index_of_interest, inplace=True)
            perf_lines_dicts = df.to_dict("records")

        m = copy.deepcopy(global_init)
        l = BaselineLoss()
        opt = Optimizer(m.parameters(), lr=LR)
        print("Pooled")
        for e in range(NUM_EPOCHS_POOLED):
            for X, y in train_pooled:
                opt.zero_grad()
                y_pred = m(X)
                loss = l(y_pred, y)
                loss.backward()
                opt.step()

        perf_dict = evaluate_model_on_tests(m, test_dls, metric)
        pooled_perf_dict = evaluate_model_on_tests(m, [test_pooled], metric)
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
        df.to_csv(NAME_RESULTS_FILE, index=False)

    # Local Baselines
    for i in range(NUM_CLIENTS):
        # we only launch training if it's not finished already
        index_of_interest = df.loc[df["Method"] == f"Local {i}"].index
        # an experiment is finished if there are num_clients + 1 rows
        if len(index_of_interest) < (NUM_CLIENTS + 1):
            # dealing with edge case that shouldn't happen
            # If some of the rows are there but not all of them we redo the experiments
            if len(index_of_interest) > 0:
                df.drop(index_of_interest, inplace=True)
                perf_lines_dicts = df.to_dict("records")
            m = copy.deepcopy(global_init)
            l = BaselineLoss()
            opt = Optimizer(m.parameters(), lr=LR)
            print("Local " + str(i))
            for e in range(NUM_EPOCHS_POOLED):
                for X, y in training_dls[i]:
                    opt.zero_grad()
                    y_pred = m(X)
                    loss = l(y_pred, y)
                    loss.backward()
                    opt.step()
            perf_dict = evaluate_model_on_tests(m, test_dls, metric)
            pooled_perf_dict = evaluate_model_on_tests(m, [test_pooled], metric)
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
            # We update csv and save it when the results are there
            df = pd.DataFrame.from_dict(perf_lines_dicts)
            df.to_csv(NAME_RESULTS_FILE, index=False)

    # Strategies
    for num_updates in [50, 100]:
        for sname in strategy_names:
            # Base arguments
            m = copy.deepcopy(global_init)
            l = BaselineLoss()
            args = {
                "training_dataloaders": training_dls,
                "model": m,
                "loss": l,
                "optimizer_class": Optimizer,
                "learning_rate": LR,
                "num_updates": num_updates,
                "nrounds": get_nb_max_rounds(num_updates),
            }
            strategy_specific_hp_dict = strategy_specific_hp_dicts[sname]
            # Overwriting arguments with strategy specific arguments
            for k, v in strategy_specific_hp_dict.items():
                args[k] = v
            # we only launch training if it's not finished already maybe FL
            # hyperparameters need to be tuned
            hyperparameters = {}
            for k in columns_names:
                if k in args:
                    hyperparameters[k] = str(args[k])
                else:
                    hyperparameters[k] = None

            index_of_interest = df.loc[
                (df["Method"] == (sname + str(num_updates)))
                & (df[list(hyperparameters)] == pd.Series(hyperparameters)).all(axis=1)
            ].index
            # an experiment is finished if there are num_clients + 1 rows
            if len(index_of_interest) < (NUM_CLIENTS + 1):
                # dealing with edge case that shouldn't happen
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
                df.to_csv(NAME_RESULTS_FILE, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--GPU",
        type=int,
        default=0,
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
