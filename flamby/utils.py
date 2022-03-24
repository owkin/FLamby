import os
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

import flamby.datasets as datasets

torch.manual_seed(42)
torch.use_deterministic_algorithms(True)


def evaluate_model_on_tests(model, test_dataloaders, metric, use_gpu=True):
    """This function takes a pytorch model and evaluate it on a list of\
    dataloaders using the provided metric function.
    Parameters
    ----------
    model: torch.nn.Module,
        A trained model that can forward the test_dataloaders outputs
    test_dataloaders: List[torch.utils.data.DataLoader]
        A list of torch dataloaders
    metric: callable,
        A function with the following signature:\
            (y_true: np.ndarray, y_pred: np.ndarray) -> scalar
    use_gpu: bool, optional,
        Whether or not to perform computations on GPU if available. \
        Defaults to True.
    Returns
    -------
    dict
        A dictionnary with keys client_test_{0} to \
        client_test_{len(test_dataloaders) - 1} and associated scalar metrics \
        as leaves.
    """
    results_dict = {}
    if torch.cuda.is_available() and use_gpu:
        model = model.cuda()
    with torch.inference_mode():
        for i in tqdm(range(len(test_dataloaders))):
            test_dataloader_iterator = iter(test_dataloaders[i])
            y_pred_final = []
            y_true_final = []
            for (X, y) in test_dataloader_iterator:
                if torch.cuda.is_available() and use_gpu:
                    X = X.cuda()
                    y = y.cuda()
                y_pred = model(X).detach().cpu()
                y = y.detach().cpu()
                y_pred_final.append(y_pred.numpy())
                y_true_final.append(y.numpy())
            y_true_final = np.vstack(y_true_final)
            y_pred_final = np.vstack(y_pred_final)
            results_dict[f"client_test_{i}"] = metric(y_true_final, y_pred_final)
    return results_dict


def read_config(config_file):
    """Read a config file in YAML.

    Parameters
    ----------
    config_file : str
        Path towards the con fig file in YAML.

    Returns
    -------
    dict
        The parsed config
    Raises
    ------
    FileNotFoundError
        If the config file does not exist
    """
    if not (os.path.exists(config_file)):
        raise FileNotFoundError("Could not find the config to read.")
    with open(config_file, "r") as file:
        dict = yaml.load(file, Loader=yaml.FullLoader)
    return dict


def get_config_file_path(debug, dataset_name):
    """Get the config_file path in real or debug mode.

    Parameters
    ----------
    debug : bool
       The mode in which we download the dataset.
    dataset_name: str
        The name of the dataset to get the config from.

    Returns
    -------
    str
        The path towards the config file.
    """
    assert dataset_name in [
        "fed_camelyon16",
        "fed_isic2019",
        "fed_lidc_idri",
    ], f"Dataset name {dataset_name} not valid."
    config_file_name = (
        "dataset_location_debug.yaml" if debug else "dataset_location.yaml"
    )
    datasets_dir = str(Path(os.path.realpath(datasets.__file__)).parent.resolve())
    path_to_config_file_folder = os.path.join(
        datasets_dir, dataset_name, "dataset_creation_scripts"
    )
    config_file = os.path.join(path_to_config_file_folder, config_file_name)
    return config_file


def create_config(output_folder, debug, dataset_name="fed_camelyon16"):
    """Create or modify config file by writing the absolute path of \
        output_folder in its dataset_path key.

    Parameters
    ----------
    output_folder : str
        The folder where the dataset will be downloaded.
    debug : bool
        Whether or not we are in debug mode.
    dataset_name: str
        The name of the dataset to get the config from.

    Returns
    -------
    Tuple(dict, str)
        The parsed config and the path to the file written on disk.
    Raises
    ------
    ValueError
        If output_folder is not a directory.
    """
    if not (os.path.isdir(output_folder)):
        raise ValueError(f"{output_folder} is not recognized as a folder")

    config_file = get_config_file_path(debug, dataset_name)

    if not (os.path.exists(config_file)):
        dataset_path = os.path.realpath(output_folder)
        dict = {
            "dataset_path": dataset_path,
            "download_complete": False,
            "preprocessing_complete": False,
        }

        with open(config_file, "w") as file:
            yaml.dump(dict, file)
    else:
        dict = read_config(config_file)

    return dict, config_file


def write_value_in_config(config_file, key, value):
    """Update config_file by modifying one of its key with its new value.

    Parameters
    ----------
    config_file : str
        Path towards a config file
    key : str
        A key belonging to download_complete, preprocessing_complete, dataset_path
    value : Union[bool, str]
        The value to write for the key field.
    Raises
    ------
    ValueError
        If the config file does not exist.
    """
    if not (os.path.exists(config_file)):
        raise FileNotFoundError(
            "The config file doesn't exist. \
            Please create the config file before updating it."
        )
    dict = read_config(config_file)
    dict[key] = value
    with open(config_file, "w") as file:
        yaml.dump(dict, file)


def check_dataset_from_config(dataset_name, debug):
    """Verify that the dataset is ready to be used by reading info from the config
    files.

    Parameters
    ----------
    dataset_name: str
        The name of the dataset to check
    debug : bool
        Whether to use the debug dataset or not.
    Returns
    -------
    dict
        The parsed config.
    Raises
    ------
    ValueError
        The dataset download or preprocessing did not finish.
    """
    try:
        dict = read_config(get_config_file_path(debug, dataset_name))
    except FileNotFoundError:
        if debug:
            raise ValueError(
                f"The dataset was not downloaded, config file \
                not found for debug mode. Please refer to \
                the download instructions inside \
                FLamby/flamby/datasets/{dataset_name}/README.md"
            )
        else:
            debug = True
            print(
                "WARNING USING DEBUG MODE DATASET EVEN THOUGH DEBUG WAS \
                SET TO FALSE, COULD NOT FIND NON DEBUG DATASET CONFIG FILE"
            )
            try:
                dict = read_config(get_config_file_path(debug, dataset_name))
            except FileNotFoundError:
                raise ValueError(
                    f"The dataset was not downloaded, config file\
                not found for either normal or debug mode. Please refer to \
                the download instructions inside \
                FLamby/flamby/datasets/{dataset_name}/README.md"
                )
    if not (dict["download_complete"]):
        raise ValueError(
            "It seems the dataset was only partially downloaded, \
            restart the download script to finish the download."
        )
    if not (dict["preprocessing_complete"]):
        raise ValueError(
            "It seems the preprocessing for this dataset is not \
             yet finished please run the appropriate preprocessing scripts before use"
        )
    return dict
