import argparse

from flamby.utils import get_config_file_path, write_value_in_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--new-path",
        type=str,
        help="The new path where the dataset has been oved to.",
        required=True,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="whether or not to update the config fro debug mode or the real one.",
    )
    args = parser.parse_args()
    path_to_config_file = get_config_file_path("fed_ixi", args.debug)
    write_value_in_config(path_to_config_file, "dataset_path", args.new_path)
