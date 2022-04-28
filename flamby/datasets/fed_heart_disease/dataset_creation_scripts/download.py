import argparse
import io
import os
import re
import sys
import wget
import hashlib
from pathlib import Path


import pandas as pd
from tqdm import tqdm

from flamby.utils import create_config, write_value_in_config


def main(output_folder, debug=False):
    """Download the heart disease dataset.

    Parameters
    ----------
    output_folder : str
        The folder where to download the dataset.
    """

    # location of the files in the UCI archive
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    centers = [
        "cleveland",
        "hungarian",
        "switzerland",
        "va"
    ]
    md5_hashes = [
        "2d91a8ff69cfd9616aa47b59d6f843db",
        "22e96bee155b5973568101c93b3705f6",
        "9a87f7577310b3917730d06ba9349e20",
        "4249d03ca7711e84f4444768c9426170"
    ]

    os.makedirs(output_folder, exist_ok=True)

    output_filenames = []

     # Creating config file with path to dataset
    dict, config_file = create_config(output_folder, debug, "fed_heart_disease")
    if dict["download_complete"]:
        print("You have already downloaded the heart disease dataset, aborting.")
        sys.exit()
    dataset_path = dict["dataset_path"]

    # get status of download
    downloaded_status_file_path = os.path.join(
        output_folder, "download_status_file.csv"
    )
    if not (os.path.exists(downloaded_status_file_path)):
        downloaded_status_file = pd.DataFrame()
        downloaded_status_file["Status"] = ["Not found"] * 4

        downloaded_status_file.to_csv(
            downloaded_status_file_path, index=False
        )
    else:
        downloaded_status_file = pd.read_csv(downloaded_status_file_path)

    # for each center, check if downloaded and download if necessary
    for i, center in enumerate(centers):
        file_status_ok = downloaded_status_file.loc[i, "Status"] == "Downloaded"

        if not file_status_ok:
            fname = wget.download(base_url + "processed." + center + ".data", out=output_folder)

            hash_md5 = hashlib.md5()
            with open(fname, "rb") as f:
                hash_md5.update(f.read())

                if hash_md5.hexdigest() == md5_hashes[i]:
                    downloaded_status_file.loc[i, "Status"] = "Downloaded"
                else:
                    downloaded_status_file.loc[i, "Status"] = "Corrupted"

        print()

    # We assert we have everything and write it
    if all((downloaded_status_file["Status"] == "Downloaded").tolist()):
        write_value_in_config(config_file, "download_complete", True)
        write_value_in_config(config_file, "preprocessing_complete", True)
        downloaded_status_file.to_csv(
            downloaded_status_file_path, index=False
        )

    else:
        print("Downloading failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-folder",
        type=str,
        help="Where to store the downloaded data.",
        required=True,
    )

    args = parser.parse_args()
    main(args.output_folder)
