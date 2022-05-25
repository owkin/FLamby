import json
from pathlib import Path

import torch  # noqa:F401  # necessary for importing optimizer


def check_config(config_path):
    config = json.loads(Path(config_path).read_text())
    # ensure that dataset exists
    try:
        # try importing the dataset from the config file
        getattr(
            __import__("flamby.datasets", fromlist=[config["dataset"]]),
            config["dataset"],
        )
    except AttributeError:
        raise AttributeError(
            f"Dataset {config['dataset']} has not been found in flamby.datasets."
            "Please ensure that the spelling is correct."
        )

    # ensure that the strategies exist
    for strategy in config["strategies"]:
        try:
            # try importing the strategy from the config file
            getattr(__import__("flamby.strategies", fromlist=[strategy]), strategy)
        except AttributeError:
            raise AttributeError(
                f"Strategy {strategy} has not been found in flamby.strategies."
                "Please ensure that the spelling is correct."
            )
        if "optimizer_class" in config["strategies"][strategy].keys():
            # ensure that optimizer (if any) comes from the torch library
            if not config["strategies"][strategy]["optimizer_class"].startswith(
                "torch."
            ):
                raise ValueError("Optimizer must be from torch")

    # ensure that the results file exists if not create it
    results_file = Path(config["results_file"])

    if not results_file.suffix == ".csv":
        results_file.with_suffix(".csv")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    return config


def get_dataset_args(
    config,
    params=[
        "BATCH_SIZE",
        "LR",
        "NUM_CLIENTS",
        "NUM_EPOCHS_POOLED",
        "Baseline",
        "BaselineLoss",
    ],
):
    param_list = []
    for param in params:
        try:
            p = getattr(
                __import__(f"flamby.datasets.{config['dataset']}", fromlist=param),
                param,
            )
        except AttributeError:
            p = None
        param_list.append(p)

    fed_dataset_name = config["dataset"].split("_")
    fed_dataset_name = "".join([name.capitalize() for name in fed_dataset_name])
    if fed_dataset_name == "FedIxi":
        fed_dataset_name = "FedIXITiny"

    fed_dataset = getattr(
        __import__(f"flamby.datasets.{config['dataset']}", fromlist=fed_dataset_name),
        fed_dataset_name,
    )
    return config["dataset"], fed_dataset, param_list


def get_strategies(config, learning_rate=None, args={}):

    strategies = config["strategies"]
    if args and any([v is not None for k, v in args.items()]):
        if args["strategy"] is not None:
            strategies = {
                args["strategy"]: {
                    "optimizer_class": args["optimizer_class"],
                    "learning_rate": args["learning_rate"]
                    if args["learning_rate"] is not None
                    else learning_rate,
                }
            }
            if args["mu"] is not None:
                assert args["strategy"] == "FedProx"
                strategies["FedProx"]["mu"] = args["mu"]
            if args["server_learning_rate"] is not None:
                assert args["strategy"] in [
                    "Scaffold",
                    "FedAdam",
                    "FedYogi",
                    "FedAdagrad",
                ]
                strategies[args["strategy"]]["server_learning_rate"] = args[
                    "server_learning_rate"
                ]
            if args["strategy"] == "Cyclic":
                strategies[args["strategy"]]["deterministic_cycle"] = args.deterministic

    for strategy in strategies.keys():
        if "optimizer_class" in strategies[strategy].keys():
            # have optimizer as a collable param and not a string
            strategies[strategy]["optimizer_class"] = eval(
                strategies[strategy]["optimizer_class"]
            )
        if "learning_rate_scaler" in strategies[strategy].keys():
            if learning_rate is None:
                raise ValueError("Learning rate is not defined. Please define it")
            # calculate learning rate
            strategies[strategy]["learning_rate"] = (
                learning_rate / strategies[strategy]["learning_rate_scaler"]
            )
            strategies[strategy].pop("learning_rate_scaler")
    return strategies


def get_results_file(config):
    return Path(config["results_file"])


if __name__ == "__main__":
    get_strategies()
    # check_config(config)
