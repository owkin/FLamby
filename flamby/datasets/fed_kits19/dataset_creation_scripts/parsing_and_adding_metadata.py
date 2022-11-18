#    Copyright 2020 Division of Medical Image Computing, German Cancer Research
# Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Part of this file comes from https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunet
# See flamby/datasets/fed_kits19/dataset_creation_scripts/LICENSE/README.md for more
# information


import argparse
import csv
import os
import shutil
from collections import defaultdict

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p,
    save_json,
    subfolders,
)

from flamby.datasets.fed_kits19.dataset_creation_scripts.utils import (
    set_environment_variables,
)
from flamby.utils import get_config_file_path, read_config, write_value_in_config


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Specify if debug mode (True) or not (False)",
    )
    args = parser.parse_args()
    return args


def read_csv_file(csv_file="../metadata/anony_sites.csv"):
    print(" Reading kits19 Meta Data ...")
    columns = defaultdict(list)  # each value in each column is appended to a list

    with open(csv_file) as f:
        reader = csv.DictReader(f)  # read rows into a dictionary format
        for row in reader:  # read a row as {column1: value1, column2: value2,...}
            for (k, v) in row.items():  # go over each column name and value
                if k == "case":
                    columns[k].append(v)  # append the value into the appropriate list
                    # based on column name k
                else:
                    columns[k].append(int(v))

    case_ids = columns["case"]
    site_ids = columns["site_id"]

    case_ids_array = np.array(site_ids)
    unique_hospital_IDs = np.unique(case_ids_array)

    # Now apply Thresholding
    thresholded_case_ids = None
    # case_ids_array.shape

    train_case_ids = case_ids[0:210]
    train_site_ids = site_ids[0:210]

    for ID in range(0, 89):
        client_ids = np.where(np.array(train_site_ids) == ID)[0]
        if len(client_ids) >= 10:
            client_data_idxx = np.array([train_case_ids[i] for i in client_ids])
            if thresholded_case_ids is None:
                thresholded_case_ids = client_data_idxx
            else:
                thresholded_case_ids = np.concatenate(
                    (thresholded_case_ids, client_data_idxx), axis=0
                )

    print(" Creating Thresholded Data's metadata file ")
    with open("../metadata/thresholded_sites.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["case_ids", "site_ids", "train_test_split", "train_test_split_silo"]
        )
        silo_count = 0
        for ID in range(0, 89):
            client_ids = np.where(np.array(train_site_ids) == ID)[0]
            if len(client_ids) >= 10:
                client_data_idxx = np.array([train_case_ids[i] for i in client_ids])
                data_length = len(client_data_idxx)
                train_ids = int(0.8 * data_length)
                for i in client_data_idxx[:train_ids]:
                    writer.writerow([i, silo_count, "train", "train_" + str(silo_count)])
                for i in client_data_idxx[train_ids:]:
                    writer.writerow([i, silo_count, "test", "test_" + str(silo_count)])

                if thresholded_case_ids is None:
                    thresholded_case_ids = client_data_idxx
                else:
                    thresholded_case_ids = np.concatenate(
                        (thresholded_case_ids, client_data_idxx), axis=0
                    )
                silo_count += 1

    return case_ids, site_ids[0:210], unique_hospital_IDs, thresholded_case_ids.tolist()


if __name__ == "__main__":
    """
    This is the KiTS dataset after Nick fixed all the labels that had errors.
    Downloaded on Jan 6th 2020
    """

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    # set_environment_variables should be called before importing nnunet
    set_environment_variables(args.debug)
    from nnunet.paths import base, nnUNet_raw_data

    path_to_config_file = get_config_file_path("fed_kits19", debug=args.debug)
    dict = read_config(path_to_config_file)
    base = base + "data"
    task_id = 64
    task_name = "KiTS_labelsFixed"

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    train_patient_names = []
    test_patient_names = []
    all_cases = subfolders(base, join=False)
    case_ids, site_ids, unique_hospital_ids, thresholded_ids = read_csv_file()

    print(thresholded_ids)
    if args.debug:
        train_patients = thresholded_ids[:25]
        test_patients = all_cases[210:211]  # we do not need the test data
    else:
        train_patients = thresholded_ids
        test_patients = all_cases[210:211]  # we do not need the test data

    for p in train_patients:
        curr = join(base, p)
        label_file = join(curr, "segmentation.nii.gz")
        image_file = join(curr, "imaging.nii.gz")
        shutil.copy(image_file, join(imagestr, p + "_0000.nii.gz"))
        shutil.copy(label_file, join(labelstr, p + ".nii.gz"))
        train_patient_names.append(p)

    for p in test_patients:
        curr = join(base, p)
        image_file = join(curr, "imaging.nii.gz")
        shutil.copy(image_file, join(imagests, p + "_0000.nii.gz"))
        test_patient_names.append(p)

    json_dict = {}
    json_dict["name"] = "KiTS"
    json_dict["description"] = "kidney and kidney tumor segmentation"
    json_dict["tensorImageSize"] = "4D"
    json_dict["reference"] = "KiTS data for nnunet_library"
    json_dict["licence"] = ""
    json_dict["release"] = "0.0"
    json_dict["modality"] = {"0": "CT"}
    json_dict["labels"] = {"0": "background", "1": "Kidney", "2": "Tumor"}

    json_dict["numTraining"] = len(train_patient_names)
    json_dict["numTest"] = len(test_patient_names)
    json_dict["training"] = [
        {
            "image": "./imagesTr/%s.nii.gz" % i.split("/")[-1],
            "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1],
        }
        for i in train_patient_names
    ]
    json_dict["test"] = [
        "./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in test_patient_names
    ]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))
    write_value_in_config(path_to_config_file, "download_complete", True)
