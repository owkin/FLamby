import sys

from flamby.datasets.fed_kits19.dataset_creation_scripts.nnunet_library.set_environment_variables import (
    set_environment_variables,
)
from flamby.utils import get_config_file_path, write_value_in_config

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Specify if debug mode (True) or not (False)",
    )

    # The three following argument are added as there will be used by
    # nnunet.experiment_planning.nnUNet_plan_and_preprocess.main()
    parser.add_argument(
        "-t",
        "--task_ids",
        nargs="+",
        help="List of integers belonging to the task ids you wish to run"
        " experiment planning and preprocessing for. Each of these "
        "ids must, have a matching folder 'TaskXXX_' in the raw "
        "data folder",
    )
    parser.add_argument(
        "-tl",
        type=int,
        required=False,
        default=8,
        help="Number of processes used for preprocessing the low resolution data for the 3D low "
        "resolution U-Net. This can be larger than -tf. Don't overdo it or you will run out of "
        "RAM",
    )
    parser.add_argument(
        "-tf",
        type=int,
        required=False,
        default=8,
        help="Number of processes used for preprocessing the full resolution data of the 2D U-Net and "
        "3D U-Net. Don't overdo it or you will run out of RAM",
    )
    args = parser.parse_args()

    # set_environment_variables should be called before importing nnunet
    set_environment_variables(args.debug)
    from nnunet.experiment_planning.nnUNet_plan_and_preprocess import main

    # We need to remove --debug from sys.argv as it is not listed in the CLI
    # of nnunet.experiment_planning.nnUNet_plan_and_preprocess.main()
    if "--debug" in sys.argv:
        sys.argv.remove("--debug")

    main()
    path_to_config_file = get_config_file_path("fed_kits19", False)
    write_value_in_config(path_to_config_file, "preprocessing_complete", True)
