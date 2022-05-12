import argparse
import requests
import shutil
import zipfile
import os
import sys

from pathlib import Path
from tqdm import tqdm
from flamby.utils import create_config, write_value_in_config

# IXI Tiny

TINY_URL = 'https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/7kd5wj7v7p-1.zip'

def dl_ixi_tiny(output_folder, debug=False):
    """
    Download the IXI Tiny dataset.

    Parameters
        ----------
        output_folder : str
            The folder where to download the dataset.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Creating config file with path to dataset
    dict, config_file = create_config(output_folder, debug, "fed_ixi")
    if dict["download_complete"]:
        print("You have already downloaded the IXI dataset, aborting.")
        sys.exit()

    img_zip_archive_name = TINY_URL.split('/')[-1]
    img_archive_path = Path(output_folder).joinpath(img_zip_archive_name)

    with requests.get(TINY_URL, stream=True) as response:
        # Raise error if not 200
        response.raise_for_status()
        file_size = int(response.headers.get('Content-Length', 0))
        desc = '(Unknown total file size)' if file_size == 0 else ''
        print(f'Downloading to {img_archive_path}')
        with tqdm.wrapattr(response.raw, 'read', total=file_size, desc=desc) as r_raw:
            with open(img_archive_path, 'wb') as f:
                shutil.copyfileobj(r_raw, f)
    
    # extraction
    print(f'Extracting to {output_folder}')
    with zipfile.ZipFile(f'{img_archive_path}', 'r') as zip_ref:
        for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
            zip_ref.extract(member=file, path=output_folder)
    
    write_value_in_config(config_file, "download_complete", True)
    write_value_in_config(config_file, "preprocessing_complete", True)
    

if __name__ == "__main__":
    parser_tiny = argparse.ArgumentParser()

    parser_tiny.add_argument(
        "-o",
        "--output-folder",
        type=str,
        help="Where to store the downloaded files.",
        required=True
    )

    parser_tiny.add_argument(
        "--debug", action="store_true", help="Allows a fast testing."
    )

    args = parser_tiny.parse_args()
    dl_ixi_tiny(args.output_folder, args.debug)