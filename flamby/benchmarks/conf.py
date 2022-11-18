import inspect
import json
from pathlib import Path

import torch  # noqa:F401  # necessary for importing optimizer

import flamby


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
    dataset_name,
    params=[
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
    ],
):
    """Get dataset spepcific handles

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to use.
    params : list, optional
        All named pparameters to be fetched, by default
        [ "BATCH_SIZE", "LR", "NUM_CLIENTS", "NUM_EPOCHS_POOLED", "Baseline",
        "BaselineLoss", "Optimizer", "get_nb_max_rounds", "metric",
        "collate_fn", ]

    Returns
    -------
    tuple(str, torch.utils.data.Dataset, list)
        _description_
    """
    # We first get all parameters excluding datasets
    param_list = []
    for param in params:
        try:
            p = getattr(
                __import__(f"flamby.datasets.{dataset_name}", fromlist=param), param
            )
        except AttributeError:
            p = None
        param_list.append(p)

    fed_dataset_name = dataset_name.split("_")
    fed_dataset_name = "".join([name.capitalize() for name in fed_dataset_name])

    if fed_dataset_name == "FedIxi":
        fed_dataset_name = "FedIXITiny"

    fed_dataset = getattr(
        __import__(f"flamby.datasets.{dataset_name}", fromlist=fed_dataset_name),
        fed_dataset_name,
    )
    return fed_dataset, param_list


def get_strategies(config, learning_rate=None, args={}):
    """Parse the config to extract strategies and hyperparameters.
    Parameters
    ----------
    config : dict
        The config dict.
    learning_rate : float
        The learning rate to use, by default None
    args : dict, optional
        The dict given by the CLI, by default {} if given will supersede the
        config.

    Returns
    -------
    dict
        dict with all strategies and their hyperparameters.

    Raises
    ------
    ValueError
        Some parameter are incorrect.
    """
    if args["strategy"] is not None:
        strategies = {args["strategy"]: {}}
        for k, v in args.items():
            if k in [
                "mu",
                "server_learning_rate",
                "learning_rate",
                "num_fine_tuning_steps",
                "optimizer_class",
                "deterministic",
                "tau",
                "beta1",
                "beta2",
                "dp_target_epsilon",
                "dp_target_delta",
                "dp_max_grad_norm",
            ] and (v is not None):
                strategies[args["strategy"]][k] = v
        if args["strategy"] != "Cyclic":
            strategies[args["strategy"]].pop("deterministic")
        else:
            strategies[args["strategy"]]["deterministic_cycle"] = strategies[
                args["strategy"]
            ].pop("deterministic")

    else:
        strategies = config["strategies"]

    for sname, sparams in strategies.items():
        possible_parameters = dict(
            inspect.signature(getattr(flamby.strategies, sname)).parameters
        )
        non_compatible_parameters = [
            param
            for param, _ in sparams.items()
            if not ((param in possible_parameters) or (param == "learning_rate_scaler"))
        ]
        assert (
            len(non_compatible_parameters) == 0
        ), f"The parameter.s {non_compatible_parameters} is/are not"
        "compatible with the strategy's signature. "
        f"Please check the {sname} strategy documentation."

        # We occasionally apply the scaler
        if ("learning_rate" in sparams) and ("learning_rate_scaler" in sparams):
            raise ValueError(
                "Cannot provide both a leraning rate and a learning rate scaler."
            )
        elif "learning_rate" not in sparams:
            scaler = (
                1.0
                if not ("learning_rate_scaler" in sparams)
                else sparams.pop("learning_rate_scaler")
            )
            strategies[sname]["learning_rate"] = learning_rate * float(scaler)

        if "optimizer_class" in sparams:
            strategies[sname]["optimizer_class"] = eval(sparams.pop("optimizer_class"))

        if (sname == "FedProx") and "mu" not in sparams:
            raise ValueError("If using FedProx you should provide a value for mu.")

        if (sname == "FedAvgFineTuning") and "num_fine_tuning_steps" not in sparams:
            raise ValueError(
                "If using FedAvgFineTuning you should provide a value"
                "for num_fine_tuning_steps (number of fine tuning step)."
            )

    return strategies


def get_results_file(config, path=None):
    if path is None:
        return Path(config["results_file"])
    else:
        return Path(path)


if __name__ == "__main__":
    get_strategies()
    # check_config(config)
