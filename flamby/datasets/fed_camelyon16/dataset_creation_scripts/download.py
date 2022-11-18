import argparse
import io
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from google_client import create_service
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from tqdm import tqdm

import flamby.datasets.fed_camelyon16.dataset_creation_scripts as dl_module
from flamby.utils import accept_license, create_config, write_value_in_config

SLIDES_LINKS_FOLDER = os.path.dirname(dl_module.__file__)


def main(path_to_secret, output_folder, port=6006, debug=False):
    """This function will download the slides from Camelyon16's Google Drive.

    Parameters
    ----------
    path_to_secret : str,
        The path towards the user's Google drive API Oauth2 secret. See README for
        how to download it.
    output_folder : str
        The folder where to download the slides. It should be large enough.
    port : int, optional
        The port by which to redirect the API requests, by default 6006
    debug : bool, optional
        Whether or not to download only a few images to check feasibility,
        by default False
    """
    accept_license("https://camelyon17.grand-challenge.org/Data/", "fed_camelyon16")
    if debug:
        print(
            "WARNING YOU ARE DOWNLOADING ONLY PART OF THE DATASET YOU ARE"
            " IN DEBUG MODE !"
        )
    train_df = pd.read_csv(
        str(Path(SLIDES_LINKS_FOLDER) / Path("training_slides_links_drive.csv"))
    )
    test_df = pd.read_csv(
        str(Path(SLIDES_LINKS_FOLDER) / Path("test_slides_links_drive.csv"))
    )
    if debug:
        train_df = train_df.iloc[:5]
        test_df = test_df.iloc[:5]
    os.makedirs(output_folder, exist_ok=True)

    # This file asserts that all files were indeed properly downloaded without corruption
    downloaded_image_status_file_name = (
        "images_download_status_file_debug.csv"
        if debug
        else "images_download_status_file.csv"
    )
    downloaded_images_status_file_path = os.path.join(
        output_folder, downloaded_image_status_file_name
    )
    if not (os.path.exists(downloaded_images_status_file_path)):
        downloaded_images_status_file = pd.DataFrame()
        downloaded_images_status_file["Status"] = ["Not found"] * (
            len(train_df.index) + len(test_df.index)
        )
        downloaded_images_status_file["Slide"] = None
        total_size = len(train_df.index) + len(test_df.index)
        train_idxs = np.arange(0, len(train_df.index))
        test_idxs = np.arange(len(train_df.index), total_size)
        downloaded_images_status_file.Slide.iloc[train_idxs] = train_df["name"]
        downloaded_images_status_file.Slide.iloc[test_idxs] = test_df["name"]
        downloaded_images_status_file.to_csv(
            downloaded_images_status_file_path, index=False
        )
    else:
        downloaded_images_status_file = pd.read_csv(downloaded_images_status_file_path)

    # Creating config file with path to dataset
    dict, config_file = create_config(output_folder, debug, "fed_camelyon16")
    if dict["download_complete"]:
        print("You have already downloaded the slides, aborting.")
        sys.exit()
    dataset_path = dict["dataset_path"]

    drive_service = create_service(
        path_to_secret,
        "drive",
        "v3",
        ["https://www.googleapis.com/auth/drive"],
        port=port,
    )
    regex = "(?<=https://drive.google.com/file/d/)[a-zA-Z0-9]+"
    # Resourcekey is now mandatory (credit @Kris in:
    # https://stackoverflow.com/questions/71343002/
    # downloading-files-from-public-google-drive-in-python-scoping-issues)
    regex_rkey = "(?<=resourcekey=).+"
    for current_df in [train_df, test_df]:
        for i in tqdm(range(len(current_df.index))):
            row = current_df.iloc[i]
            slide_path = os.path.join(output_folder, row["name"])
            file_status_ok = (
                downloaded_images_status_file.loc[
                    downloaded_images_status_file["Slide"] == row["name"], "Status"
                ]
                == "Downloaded"
            ).item()
            if file_status_ok:
                continue
            else:
                if os.path.exists(slide_path):
                    # We assume the file is corrupted and delete it
                    os.remove(slide_path)

            # Getting the URLs
            url = row["link"]
            file_id = re.search(regex, url)[0]
            resource_key = re.search(regex_rkey, url)[0]
            request = drive_service.files().get_media(fileId=file_id)
            request.headers["X-Goog-Drive-Resource-Keys"] = f"{file_id}/{resource_key}"

            name = row["name"]
            print(f"Downloading slide {name} into {os.path.realpath(dataset_path)}")
            # Adding a try/except clause
            try:
                fh = io.FileIO(slide_path, mode="wb")
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                with tqdm(total=100) as pbar:
                    while done is False:
                        status, done = downloader.next_chunk()
                        pbar.update(int(status.progress() * 100) - pbar.n)
                # Only if we reach 100% completion we count the file as downloaded
                downloaded_images_status_file.loc[
                    downloaded_images_status_file["Slide"] == row["name"], "Status"
                ] = "Downloaded"
            except HttpError:
                # if there was an error (e.g. quota not reached), we record it
                # and we move on to the next
                downloaded_images_status_file.loc[
                    downloaded_images_status_file["Slide"] == row["name"], "Status"
                ] = "Error during download"

            downloaded_images_status_file.to_csv(
                downloaded_images_status_file_path, index=False
            )

    # We assert we have everything and write it
    if all((downloaded_images_status_file["Status"] == "Downloaded").tolist()):
        write_value_in_config(config_file, "download_complete", True)
    else:
        write_value_in_config(config_file, "download_complete", False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-to-secret",
        type=str,
        help="The path where to find the secret obtained from the OAuth2 of the\
        Google service account",
        required=True,
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        help="Where to store the downloaded tifs.",
        required=True,
    )
    parser.add_argument(
        "--port", type=int, help="The port to use for URI redirection", default=6006
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether or not to download only the first 4 slides to test the\
        download script.",
    )

    args = parser.parse_args()
    main(args.path_to_secret, args.output_folder, args.port, args.debug)
