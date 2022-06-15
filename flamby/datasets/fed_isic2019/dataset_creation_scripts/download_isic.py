import argparse
import os
import sys
import zipfile

import pandas as pd

from flamby.utils import accept_license, create_config, write_value_in_config

url_1 = (
    "https://isic-challenge-data.s3.amazonaws.com/2019/" "ISIC_2019_Training_Input.zip"
)

url_2 = (
    "https://isic-challenge-data.s3.amazonaws.com/2019/"
    "ISIC_2019_Training_Metadata.csv"
)

url_3 = (
    "https://isic-challenge-data.s3.amazonaws.com/2019/"
    "ISIC_2019_Training_GroundTruth.csv"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-folder",
    type=str,
    help="Where to store raw images, preprocessed images, ground truth, metadata, model",
    required=True,
)
args = parser.parse_args()

accept_license("https://challenge.isic-archive.com/data/", "fed_isic2019")
os.makedirs(args.output_folder, exist_ok=True)

# Creating config file with path to dataset from arguments
dict, config_file = create_config(
    output_folder=args.output_folder, debug=False, dataset_name="fed_isic2019"
)
if dict["download_complete"]:
    print("You already have downloaded the dataset. Aborting.")
    sys.exit()
data_directory = dict["dataset_path"]


dest_file_1 = os.path.join(data_directory, "ISIC_2019_Training_Input.zip")
dest_file_2 = os.path.join(data_directory, "ISIC_2019_Training_Metadata.csv")
dest_file_3 = os.path.join(data_directory, "ISIC_2019_Training_GroundTruth.csv")
dest_file_4 = os.path.join(data_directory, "ISIC_2019_Training_Metadata_FL.csv")
parent_script_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
file1 = os.path.join(parent_script_directory, "HAM10000_metadata")

# download and unzip data
os.system(f"wget {url_1} --no-check-certificate -O {dest_file_1}")
if zipfile.is_zipfile(dest_file_1):
    print("Zip file downloaded correctly")
    os.system(f"unzip {dest_file_1} -d {data_directory}")
    os.system(f"rm {dest_file_1}")
else:
    sys.exit("Zip file corrupted")
os.system(f"wget {url_2} --no-check-certificate -O {dest_file_2}")
os.system(f"wget {url_3} --no-check-certificate -O {dest_file_3}")

# create pandas dataframes
ISIC_2019_Training_Metadata = pd.read_csv(dest_file_2)
ISIC_2019_Training_GroundTruth = pd.read_csv(dest_file_3)
# keeping only image and dataset columns in the HAM10000 metadata
HAM10000_metadata = pd.read_csv(file1)
HAM10000_metadata.rename(columns={"image_id": "image"}, inplace=True)
HAM10000_metadata.drop(
    ["age", "sex", "localization", "lesion_id", "dx", "dx_type"], axis=1, inplace=True
)

# taking out images (from image set, metadata file and ground truth file)
# where datacenter is not available
for i, row in ISIC_2019_Training_Metadata.iterrows():
    if pd.isnull(row["lesion_id"]):
        image = row["image"]
        os.system("rm " + data_directory + "/ISIC_2019_Training_Input/" + image + ".jpg")
        if image != ISIC_2019_Training_GroundTruth["image"][i]:
            print("Mismatch between Metadata and Ground Truth")
        ISIC_2019_Training_GroundTruth = ISIC_2019_Training_GroundTruth.drop(i)
        ISIC_2019_Training_Metadata = ISIC_2019_Training_Metadata.drop(i)

# generating dataset field from lesion_id field in the metadata dataframe
ISIC_2019_Training_Metadata["dataset"] = ISIC_2019_Training_Metadata["lesion_id"].str[:4]

# join with HAM10000 metadata in order to expand the HAM datacenters
result = pd.merge(ISIC_2019_Training_Metadata, HAM10000_metadata, how="left", on="image")
result["dataset"] = result["dataset_x"] + result["dataset_y"].astype(str)
result.drop(["dataset_x", "dataset_y", "lesion_id"], axis=1, inplace=True)

# checking sizes and saving to csv files
print("Datacenters")
print(result["dataset"].value_counts())
print("Number of lines in Metadata", ISIC_2019_Training_Metadata.shape[0])
print("Number of lines in GroundTruth", ISIC_2019_Training_GroundTruth.shape[0])
print("Number of lines in MetadataFL", result.shape[0])
DIR = os.path.join(data_directory, "ISIC_2019_Training_Input")
N = (
    len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    - 2
)
print("Number of images", N)
result.to_csv(dest_file_4, index=False)
ISIC_2019_Training_Metadata.to_csv(dest_file_2, index=False)
ISIC_2019_Training_GroundTruth.to_csv(dest_file_3, index=False)

if N == 23247:
    print("Download OK")
    write_value_in_config(config_file, "download_complete", True)
else:
    print("Something wrong happened during the download.")
