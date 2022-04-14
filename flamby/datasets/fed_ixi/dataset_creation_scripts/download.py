import argparse
import requests
import shutil

from pathlib import Path
from tqdm import tqdm

MIRRORS = ['https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/']
DEMOGRAPHICS_FILENAME = 'IXI.xls'
ALLOWED_MODALITIES = ['T1', 'T2', 'PD', 'MRA', 'DTI']

def main(output_folder, modality, debug=False):
    if debug:
        raise NotImplementedError("Debug not implemented yet")
    if modality == 'all':
        image_urls = [MIRRORS[0] + "IXI-" + a_m + ".tar" for a_m in ALLOWED_MODALITIES]
    else:
        image_urls = [MIRRORS[0] + "IXI-" + modality.upper() + ".tar"]
        print(image_urls[0])
    
    # Make folder if it does not exist
    output_f = Path(f"./{output_folder}").resolve()
    output_f.mkdir(exist_ok=True)
    demographics_url = [MIRRORS[0] + DEMOGRAPHICS_FILENAME]  # URL EXCEL
    for file_url in image_urls + demographics_url:
        img_tarball_archive_name = file_url.split('/')[-1]
        img_archive_path = output_f.joinpath(img_tarball_archive_name)
        if img_archive_path.is_file():
            continue
        with requests.get(file_url, stream=True) as response:
            # Raise error if not 200
            response.raise_for_status()
            file_size = int(response.headers.get('Content-Length', 0))
            desc = "(Unknown total file size)" if file_size == 0 else ""
            print(f'Downloading to {img_archive_path}')
            with tqdm.wrapattr(response.raw, "read", total=file_size, desc=desc) as r_raw:
                with open(img_archive_path, 'wb') as f:
                    shutil.copyfileobj(r_raw, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-folder",
        type=str,
        help="Where to store the downloaded files.",
        required=True
    )

    parser.add_argument(
        "-m",
        "--modality",
        choices=['t1', 't2', 'pd', 'mra', 'dti', 'all'],
        type=str,
        help="What modality is downloaded. Available choices : t1, t2, pd, mra, dti, all.",
        required=True
    )

    parser.add_argument(
        "--debug", action="store_true", help="Allows a fast testing."
    )

    args = parser.parse_args()
    main(args.output_folder, args.modality, args.debug)