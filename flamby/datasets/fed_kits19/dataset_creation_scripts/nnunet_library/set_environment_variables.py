import os

from flamby.utils import get_config_file_path, read_config


def set_environment_variables(debug):
    """
    Load the config file, and set environment variables required before importing
    nnunet library
    """
    path_to_config_file = get_config_file_path("fed_kits19", debug)
    print(path_to_config_file)
    dict = read_config(path_to_config_file)

    base = dict["dataset_path"] + "/"
    os.environ["nnUNet_raw_data_base"] = base
    os.environ["nnUNet_preprocessed"] = base + "kits19_preprocessing"
    os.environ["RESULTS_FOLDER"] = base + "kits19_Results"
