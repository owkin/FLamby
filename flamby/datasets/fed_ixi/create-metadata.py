import os
from pathlib import Path

from flamby.datasets.fed_ixi.utils import (
    _create_train_test_split,
    _extract_center_name_from_filename,
    _get_id_from_filename,
)

# IXI Tiny Dataset

root = "."
root_folder = Path(root).expanduser().joinpath("IXI-Dataset")
parent_dir_name = os.path.join("IXI Sample Dataset", "7kd5wj7v7p-1", "IXI_sample")
subjects_dir = os.path.join(root_folder, parent_dir_name)

images_centers = []  # contains center of each subject: HH, Guys or IOP
images_sets = []  # train and test

subjects = [
    subject
    for subject in os.listdir(subjects_dir)
    if os.path.isdir(os.path.join(subjects_dir, subject))
]
images_centers = [_extract_center_name_from_filename(subject) for subject in subjects]

train_test_hh, train_test_guys, train_test_iop = _create_train_test_split(
    images_centers=images_centers
)

csv_file_content = "Patient ID,Manufacturer,Split"

idx_hh, idx_guys, idx_iop = 0, 0, 0

for subject in subjects:
    csv_file_content += "\n" + str(_get_id_from_filename(subject)) + ","
    center_name = _extract_center_name_from_filename(subject)
    if center_name == "HH":
        csv_file_content += f"1,{train_test_hh[idx_hh]}"
        idx_hh += 1
    elif center_name == "Guys":
        csv_file_content += f"0,{train_test_guys[idx_guys]}"
        idx_guys += 1
    else:
        csv_file_content += f"2,{train_test_iop[idx_iop]}"
        idx_iop += 1

with open("metadata_tiny.csv", "w") as file:
    file.write(csv_file_content)
