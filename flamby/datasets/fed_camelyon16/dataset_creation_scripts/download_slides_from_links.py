import argparse
import io
import os
import re
from pathlib import Path

import pandas as pd
from google import create_service
from googleapiclient.http import MediaIoBaseDownload
from tqdm import tqdm

import flamby.datasets.fed_camelyon16.dataset_creation_scripts as dl_module

SLIDES_LINKS_FOLDER = os.path.dirname(dl_module.__file__)


def main(path_to_secret, output_folder, port=6006):

    train_df = pd.read_csv(
        str(Path(SLIDES_LINKS_FOLDER) / Path("training_slides_links_drive.csv"))
    )
    test_df = pd.read_csv(
        str(Path(SLIDES_LINKS_FOLDER) / Path("test_slides_links_drive.csv"))
    )
    os.makedirs(output_folder, exist_ok=True)
    drive_service = create_service(
        path_to_secret,
        "drive",
        "v3",
        ["https://www.googleapis.com/auth/drive"],
        port=port,
    )
    regex = "(?<=https://drive.google.com/file/d/)[a-zA-Z0-9]+"
    # Resourcekey is now mandatory as well, code from https://stackoverflow.com/questions/71343002/downloading-files-from-public-google-drive-in-python-scoping-issues
    regex_rkey = "(?<=resourcekey=)[a-zA-Z0-9-]+"
    for current_df in [train_df, test_df]:
        for i in tqdm(range(len(current_df.index))):
            row = current_df.iloc[i]
            url = row["link"]
            file_id = re.search(regex, url)[0]
            resource_key = re.search(regex_rkey, url)[0]
            request = drive_service.files().get_media(fileId=file_id)
            request.headers["X-Goog-Drive-Resource-Keys"] = f"{file_id}/{resource_key}"

            slide_path = os.path.join(output_folder, row["name"])
            if os.path.exists(slide_path):
                continue
            name = row["name"]
            print(f"Downloading slide {name} into {slide_path}")
            fh = io.FileIO(slide_path, mode="wb")
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print("Download %d%%." % int(status.progress() * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-to-secret",
        type=str,
        help="The path where to find the secret obtained from the OAuth2 of the"
        """
        Google service account""",
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

    args = parser.parse_args()
    main(args.path_to_secret, args.output_folder, args.port)
