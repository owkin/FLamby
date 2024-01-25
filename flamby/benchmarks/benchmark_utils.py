import copy
import random
import time

import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine
from torch.utils.data import DataLoader as dl
from tqdm import tqdm

from flamby.utils import evaluate_model_on_tests


def set_seed(seed):
    """Set numpy, python and torch seed.
    Python seed is necessary for seeding albumentations.

    Parameters
    ----------
    seed : int
        The seed to set.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def fill_df_with_xp_results(
    df,
    perf_dict,
    hyperparams,
    method_name,
    columns_names,
    results_file,
    dump=True,
    pooled=False,
):
    """Add results to dataframe for a specific strategy with specific hyperparameters.

    Parameters
    ----------
    df : pd.DataFrame
        The Dataframe of results
    perf_dict: dict
        A dictionnary with keys being the different tests and values being the metric.
    hyperparams : dict
        The dict of hyerparameters.
    method_name : str
        The name of the training method.
    columns_names : list[str]
        The columns names in the considered dataframe.
    dump: bool
        Should it dump the dataframe to disk after having added the results.
        Defaults to True.
    pooled: bool
        If it is the pooled result we should change the name of the test to
        distinguish it from the first local test.
        Default to False.
    """

    perf_lines_dicts = df.to_dict("records")
    if pooled:
        assert (
            len(perf_dict) == 1
        ), "Your pooled perf dict has multiple keys this is impossible."
        perf_dict["Pooled Test"] = perf_dict.pop(list(perf_dict)[0])

    for k, v in perf_dict.items():
        perf_lines_dicts.append(
            prepare_dict(
                keys=columns_names,
                allow_new=True,
                Test=k,
                Metric=v,
                Method=method_name,
                # We add the hyperparameters used
                **hyperparams,
            )
        )
    # We update csv and save it when the results are there
    df = pd.DataFrame.from_dict(perf_lines_dicts)
    if dump:
        df.to_csv(results_file, index=False)
    return df


def find_xps_in_df(df, hyperparameters, sname, num_updates):
    """This function returns the index in the given dataframe where it found
    results for a given set of hyperparameters of the sname federated strategy
    with num_updates number of updates secified as a dict.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe of experiments
    hyperparameters : dict
        A dict with keys that are columns of the dataframe and values that are
        used to filter the dataframe.
    sname: str
        The name of the FL strategy to investigate.
        Should be in the following list:
        ["FedAvg", "Scaffold", "FedProx", "Cyclic", "FedAdam", "FedAdagrad",
        'FedYogi', FedAvgFineTuning,]
    num_udpates: int
        The number of batch updates used in the strategy.
    """
    # This is very ugly but this is the only way I found to accomodate float
    # and objects equality in a robust fashion
    # The non-robust version would be simpler but it doesn't handle floats well
    # index_of_interest = df.loc[
    #   (df["Method"] == (sname + str(num_updates)))
    #   & (
    #       df[list(hyperparameters)] == pd.Series(hyperparameters)
    #   ).all(axis=1)
    # ].index
    assert all(
        [e in df.columns for e in list(hyperparameters)]
    ), "Some hyperparameters provided are not included in the dataframe"
    assert sname in [
        "FedAvg",
        "Scaffold",
        "FedProx",
        "Cyclic",
        "FedAdam",
        "FedAdagrad",
        "FedYogi",
        "FedAvgFineTuning",
    ], f"Strategy name {sname} not recognized."
    found_xps = df[list(hyperparameters)]

    # Different types of data need different matching strategy
    found_xps_numerical = found_xps.select_dtypes(exclude=[object])
    col_numericals = found_xps_numerical.columns
    col_objects = [c for c in found_xps.columns if not (c in col_numericals)]

    # Special cases for boolean parameters
    if "deterministic_cycle" in found_xps_numerical.columns:
        found_xps_numerical["deterministic_cycle"] = (
            found_xps_numerical["deterministic_cycle"].fillna(0.0).astype(float)
        )

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

    # We filter on the Method we want
    bool_method = df["Method"] == (sname + str(num_updates))

    index_of_interest_1 = df.loc[pd.DataFrame(bool_numerical).all(axis=1)].index
    index_of_interest_2 = df.loc[pd.DataFrame(bool_objects).all(axis=1)].index
    index_of_interest_3 = df.loc[pd.DataFrame(bool_method).all(axis=1)].index
    index_of_interest = index_of_interest_1.intersection(
        index_of_interest_2
    ).intersection(index_of_interest_3)
    return index_of_interest


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
            (
                dl(
                    center_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    collate_fn=collate_fn,
                )
                if len(center_dataset := dataset(center=i, train=True, pooled=False))
                > 0
                else None
            )
            for i in range(num_clients)
        ]
        test_dls = [
            (
                dl(
                    center_dataset,
                    batch_size=batch_size_test,
                    shuffle=False,
                    num_workers=num_workers,
                    collate_fn=collate_fn,
                )
                if len(center_dataset := dataset(center=i, train=False, pooled=False))
                > 0
                else None
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


def get_logfile_name_from_strategy(dataset_name, sname, num_updates, args):
    """Produce exlicit logfile name from strategy num updates and args.

    Parameters
    ----------
    dataset_name : str
         The name of the dataset.
    sname : str
        The name of the strategy.
    num_updates : int
        The number of batch updates used in the strategy.
    args : dict
        The dict of hyperparameters of the strategy
    """
    basename = dataset_name + "-" + sname + f"-num-updates{num_updates}"
    for k, v in args.items():
        if k in ["learning_rate", "server_learning_rate"]:
            basename += "-" + "".join([e[0] for e in str(k).split("_")]) + str(v)
        if k in ["mu", "deterministic_cycle"]:
            basename += "-" + str(k) + str(v)
    return basename


def evaluate_model_on_local_and_pooled_tests(
    m, local_dls, pooled_dl, metric, evaluate_func, return_pred=False
):
    """Evaluate the model on a list of dataloaders and on one dataloader using
    the evaluate function given.

    Parameters
    ----------
    m : torch.nn.Module
        The model to evaluate.
    local_dls : list[torch.utils.data.DataLoader]
        The list of dataloader used for tests.
    pooled_dl : torch.utils.data.DataLoader
        The single dataloader used for test.
    metric: callable
        The metric to use for evaluation.
    evaluate_func : callable
        The function used to evaluate
    return_pred: bool
        Whether or not to return pred.

    Returns
    -------
    Tuple(dict, dict)
        Two performances dicts.
    """

    perf_dict = evaluate_func(m, local_dls, metric, return_pred=return_pred)
    pooled_perf_dict = evaluate_func(m, [pooled_dl], metric, return_pred=return_pred)

    # Very ugly tuple unpacking in case we return the predictions as well
    # in thee future the evaluation function should return a dict but there is
    # a lot of refactoring needed
    if return_pred:
        perf_dict, y_true_dict, y_pred_dict = perf_dict
        pooled_perf_dict, y_true_pooled_dict, y_pred_pooled_dict = pooled_perf_dict
    else:
        y_true_dict, y_pred_dict, y_true_pooled_dict, y_pred_pooled_dict = (
            None,
            None,
            None,
            None,
        )

    print("Per-center performance:")
    print(perf_dict)
    print("Performance on pooled test set:")
    print(pooled_perf_dict)
    return (
        perf_dict,
        pooled_perf_dict,
        y_true_dict,
        y_pred_dict,
        y_true_pooled_dict,
        y_pred_pooled_dict,
    )


def train_single_centric(
    global_init,
    train_dl,
    use_gpu,
    name,
    opt_class,
    learning_rate,
    loss_class,
    num_epochs,
    dp_target_epsilon=None,
    dp_target_delta=None,
    dp_max_grad_norm=None,
    seed=None,
):
    """Train the global_init model using train_dl and default parameters.

    Parameters
    ----------
    global_init : torch.nn.Module
        The initialized model to train.
    train_dl : torch.utils.data.DataLoader
        The dataloader to use for training.
    use_gpu : bool
        Whether or not to use the GPU.
    name : str
        The name of the method to display.
    opt_class: torch.optim
        A callable with signature (list[torch.Tensor], lr) -> torch.optim
    learning_rate: float
        The learning rate of the optimizer.
    loss_class: torch.losses._Loss
        A callable return a pytorch loss.
    num_epochs: int
         The number of epochs on which to train.
    dp_target_epsilon: float
        The target epsilon for (epsilon, delta)-differential private guarantee.
        Defaults to None.
    dp_target_delta: float
        The target delta for (epsilon, delta)-differential private guarantee.
         Defaults to None.
    dp_max_grad_norm: float
        The maximum L2 norm of per-sample gradients; used to enforce
        differential privacy. Defaults to None.

    Returns
    -------
    torch.nn.Module
       The trained model.
    """
    apply_dp = (
        (dp_target_epsilon is not None)
        and (dp_max_grad_norm is not None)
        and (dp_target_delta is not None)
    )
    if (not apply_dp) and (dp_target_epsilon is not None):
        raise ValueError("Missing argument for DP")
    if (not apply_dp) and (dp_max_grad_norm is not None):
        raise ValueError("Missing argument for DP")
    if (not apply_dp) and (dp_target_delta is not None):
        raise ValueError("Missing argument for DP")

    device = "cpu"
    model = copy.deepcopy(global_init)
    if use_gpu:
        model.cuda()
        device = "cuda"

    bloss = loss_class()
    opt = opt_class(model.parameters(), lr=learning_rate)

    if apply_dp:
        seed = seed if seed is not None else int(time.time())
        privacy_engine = PrivacyEngine()

        # put model in train mode if not already the case
        model.train()

        model, opt, train_dl = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=opt,
            data_loader=train_dl,
            epochs=num_epochs,
            target_epsilon=dp_target_epsilon,
            target_delta=dp_target_delta,
            max_grad_norm=dp_max_grad_norm,
            noise_generator=torch.Generator(device).manual_seed(seed),
        )

    grad_norm_history = []
    for _ in tqdm(range(num_epochs)):
        for X, y in train_dl:
            if use_gpu:
                # use GPU if requested and available
                X = X.cuda()
                y = y.cuda()
            opt.zero_grad()
            y_pred = model(X)
            loss = bloss(y_pred, y)
            loss.backward()
            opt.step()

            grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += torch.linalg.norm(param.grad)
            grad_norm_history.append(grad_norm)

    return model


def init_xp_plan(
    num_clients,
    nlocal,
    single_centric_baseline=None,
    strategy=None,
    compute_ensemble_perf=False,
):
    """_summary_

    Parameters
    ----------
    num_clients : int
        The number of available clients.
    nlocal : int
        The index of the chosen client.
    single_centric_baseline : str
        The single centric baseline to comute.
    strategy: str
        The strategy to compute results for.

    Returns
    -------
    dict
        A dict with the plannification of xps to do.

    Raises
    ------
    ValueError
        _description_
    """
    do_strategy = True
    do_baselines = {"Pooled": True}
    for i in range(num_clients):
        do_baselines[f"Local {i}"] = True
    # Single client baseline computation
    if single_centric_baseline is not None:
        if compute_ensemble_perf:
            print(
                "WARNING: by providing the argument single_centric_baseline"
                " you will not be able to compute ensemble performance."
            )
            compute_ensemble_perf = False
        do_baselines = {"Pooled": False}
        for i in range(num_clients):
            do_baselines[f"Local {i}"] = False
        if single_centric_baseline == "Pooled":
            do_baselines[single_centric_baseline] = True
        elif single_centric_baseline == "Local":
            assert nlocal in range(num_clients), "The client you chose does not exist"
            do_baselines[single_centric_baseline + " " + str(nlocal)] = True
        # If we do a single-centric baseline we don't do the strategies
        do_strategy = False

    # if we give a strategy we compute only the strategy and not the baselines
    if strategy is not None:
        if compute_ensemble_perf:
            print(
                "WARNING: by providing a strategy argument you will"
                " not be able to compute ensemble performance."
            )
            compute_ensemble_perf = False
        for k, _ in do_baselines.items():
            do_baselines[k] = False

    do_all_local = all([do_baselines[f"Local {i}"] for i in range(num_clients)])
    if compute_ensemble_perf and not (do_all_local):
        raise ValueError(
            "Cannot compute ensemble performance if training on only one local"
        )
    return do_baselines, do_strategy, compute_ensemble_perf


def ensemble_perf_from_predictions(
    y_true_dicts, y_pred_dicts, num_clients, metric, num_clients_test=None
):
    """_summary_

    Parameters
    ----------
    y_true_dicts : dict
        The ground truth dicts for all clients
    y_pred_dicts :dict
        The prediction array for all models and clients.
    num_clients : int
        The number of clients
    metric : callable
        (torch.Tensor, torch.Tensor) -> [0, 1.]
    num_clients_test: int
        When testing on pooled.

    Returns
    -------
    dict
        A dict with the predictions of all ensembles
    """
    print("Computing ensemble performance")
    ensemble_perf = {}
    if num_clients_test is None:
        num_clients_test = num_clients
    for testset_idx in range(num_clients_test):
        # Small safety net
        for model_idx in range(1, num_clients):
            assert (
                y_true_dicts[f"Local {0}"][f"client_test_{testset_idx}"]
                == y_true_dicts[f"Local {model_idx}"][f"client_test_{testset_idx}"]
            ).all(), "Models in the ensemble have different ground truths"

        # Since they are all the same we use the first one
        # for this specific tests as the ground truth
        ensemble_true = y_true_dicts["Local 0"][f"client_test_{testset_idx}"]

        # Accumulating predictions
        ensemble_pred = y_pred_dicts["Local 0"][f"client_test_{testset_idx}"]
        for model_idx in range(1, num_clients):
            ensemble_pred += y_pred_dicts[f"Local {model_idx}"][
                f"client_test_{testset_idx}"
            ]
        ensemble_pred /= float(num_clients)
        ensemble_perf[f"client_test_{testset_idx}"] = metric(
            ensemble_true, ensemble_pred
        )
    return ensemble_perf


def set_dataset_specific_config(
    dataset_name, compute_ensemble_perf=False, use_gpu=True
):
    """_summary_

    Parameters
    ----------
    dataset_name : _type_
        _description_
    compute_ensemble_perf: bool
        Whether or not to compute ensemble performances. Cannot be used with
        KITS or LIDC. Defaults to None.

    Returns
    -------
    _type_
        _description_
    """
    # Instantiate all train and test dataloaders required including pooled ones
    if dataset_name == "fed_lidc_idri":
        batch_size_test = 1
        from flamby.datasets.fed_lidc_idri import evaluate_dice_on_tests_by_chunks

        def evaluate_func(m, test_dls, metric, use_gpu=use_gpu, return_pred=False):
            dice_dict = evaluate_dice_on_tests_by_chunks(m, test_dls, use_gpu)
            if return_pred:
                return dice_dict, None, None
            return dice_dict

        compute_ensemble_perf = False
    elif dataset_name == "fed_kits19":
        from flamby.datasets.fed_kits19 import evaluate_dice_on_tests

        batch_size_test = 2

        def evaluate_func(m, test_dls, metric, use_gpu=use_gpu, return_pred=False):
            dice_dict = evaluate_dice_on_tests(m, test_dls, metric, use_gpu)
            if return_pred:
                return dice_dict, None, None
            return dice_dict

        compute_ensemble_perf = False

    elif dataset_name == "fed_ixi":
        batch_size_test = 1
        evaluate_func = evaluate_model_on_tests
        compute_ensemble_perf = False

    else:
        batch_size_test = None
        evaluate_func = evaluate_model_on_tests

    return evaluate_func, batch_size_test, compute_ensemble_perf
