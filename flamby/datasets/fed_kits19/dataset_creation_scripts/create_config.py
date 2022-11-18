import argparse

from flamby.utils import create_config

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
        action="store_true",
        help="whether or not to update the config fro debug mode or the real one.",
    )
    args = parser.parse_args()

    create_config(args.output_folder, args.debug, "fed_kits19")
