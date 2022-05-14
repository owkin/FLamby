import argparse

from flamby.utils import get_config_file_path, write_value_in_config, create_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_folder",
        type=str,
        help="The new path where the dataset has been oved to.",
        required=True,
    )
    parser.add_argument(
        "--debug",
        type=bool,
        action="store_false",
        help="whether or not to update the config fro debug mode or the real one.",
    )
    args = parser.parse_args()
    dict, config_file = create_config(args.output_folder, args.debug, "fed_kits19")
    #
    # path_to_config_file = get_config_file_path("fed_camelyon16", args.debug)
    # write_value_in_config(path_to_config_file, "dataset_path", args.new_path)