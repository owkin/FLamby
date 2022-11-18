import argparse

from flamby.utils import create_config, write_value_in_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, help="The path where the dataset is located", required=True
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="The name of the dataset you downloaded",
        required=True,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="whether or not to update the config fro debug mode or the real one.",
    )
    args = parser.parse_args()
    dict, config_file = create_config(args.path, args.debug, args.dataset_name)
    write_value_in_config(config_file, "dataset_path", args.path)
    write_value_in_config(config_file, "download_complete", True)
    write_value_in_config(config_file, "preprocessing_complete", True)
