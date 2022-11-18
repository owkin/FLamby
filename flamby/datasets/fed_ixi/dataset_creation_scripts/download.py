import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from flamby.utils import accept_license, create_config, write_value_in_config


def dl_ixi_tiny(output_folder, debug=False):
    """
    Download the IXI Tiny dataset.

    Parameters
        ----------
        output_folder : str
            The folder where to download the dataset.
    """
    print(
        "The IXI dataset is made available under the Creative Commons CC BY-SA \
            3.0 license.\n\
    If you use the IXI data please acknowledge the source of the IXI data, e.g.\
    the following website: https://brain-development.org/ixi-dataset/\
    IXI Tiny is derived from the same source. Acknowledge the following reference\
    on TorchIO : https://torchio.readthedocs.io/datasets.html#ixitiny\
    Pérez-García F, Sparks R, Ourselin S. TorchIO: a Python library for \
    efficient loading, preprocessing, augmentation and patch-based sampling \
    of medical images in deep learning. arXiv:2003.04696 [cs, eess, stat]. \
    2020. https://doi.org/10.48550/arXiv.2003.04696"
    )
    accept_license("https://brain-development.org/ixi-dataset/", "fed_ixi")
    os.makedirs(output_folder, exist_ok=True)

    # Creating config file with path to dataset
    dict, config_file = create_config(output_folder, debug, "fed_ixi")
    if dict["download_complete"]:
        print("You have already downloaded the IXI dataset, aborting.")
        sys.exit()
    # Deferring import to avoid circular imports
    from flamby.datasets.fed_ixi.common import DATASET_URL

    img_zip_archive_name = DATASET_URL.split("/")[-1]
    img_archive_path = Path(output_folder).joinpath(img_zip_archive_name)

    with requests.get(DATASET_URL, stream=True) as response:
        # Raise error if not 200
        response.raise_for_status()
        file_size = int(response.headers.get("Content-Length", 0))
        desc = "(Unknown total file size)" if file_size == 0 else ""
        print(f"Downloading to {img_archive_path}")
        with tqdm.wrapattr(response.raw, "read", total=file_size, desc=desc) as r_raw:
            with open(img_archive_path, "wb") as f:
                shutil.copyfileobj(r_raw, f)

    # extraction
    print(f"Extracting to {output_folder}")
    with zipfile.ZipFile(f"{img_archive_path}", "r") as zip_ref:
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
        required=True,
    )

    parser_tiny.add_argument(
        "--debug", action="store_true", help="Allows a fast testing."
    )

    args = parser_tiny.parse_args()
    dl_ixi_tiny(args.output_folder, args.debug)
