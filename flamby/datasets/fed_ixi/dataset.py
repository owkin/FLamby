import os
from pathlib import Path
from typing import Dict, Tuple
from zipfile import ZipFile

import pandas as pd
import torch
from monai.transforms import (
    AddChannel,
    AsDiscrete,
    Compose,
    NormalizeIntensity,
    Resize,
    ToTensor,
)
from torch import Tensor
from torch.utils.data import Dataset

import flamby
from flamby.datasets.fed_ixi.utils import (
    _extract_center_name_from_filename,
    _get_center_name_from_center_id,
    _get_id_from_filename,
    _load_nifti_image_and_label_by_id,
)
from flamby.utils import check_dataset_from_config


class IXITinyRaw(Dataset):
    """
    Generic interface for IXI Tiny Dataset

    Parameters
    ----------
    transform : optional
        PyTorch Transform to process the data or augment it. Default to None
    debug : bool, optional
        Default to False
    data_path: str
        If data_path is given it will ignore the config file and look for the
        dataset directly in data_path. Defaults to None.
    """

    CENTER_LABELS = {"Guys": 0, "HH": 1, "IOP": 2}

    def __init__(self, transform=None, debug=False, data_path=None):
        if data_path is None:
            dict = check_dataset_from_config("fed_ixi", debug)
            self.root_folder = Path(dict["dataset_path"])
        else:
            if not (os.path.exists(data_path)):
                raise ValueError(f"The string {data_path} is not a valid path.")
            self.root_folder = Path(data_path)

        self.metadata = pd.read_csv(
            Path(os.path.dirname(flamby.datasets.fed_ixi.__file__))
            / Path("metadata")
            / Path("metadata_tiny.csv"),
            index_col="Patient ID",
        )

        # pd.read_csv('./metadata/metadata_tiny.csv')
        self.common_shape = (48, 60, 48)
        self.transform = transform
        self.modality = "T1"

        # Download of the ixi tiny must be completed and extracted to run this part
        # Deferring the import to avoid circular imports
        from flamby.datasets.fed_ixi.common import DATASET_URL, FOLDER

        self.image_url = DATASET_URL
        self.parent_folder = FOLDER

        self.parent_dir_name = os.path.join(self.parent_folder, "IXI_sample")
        self.subjects_dir = os.path.join(self.root_folder, self.parent_dir_name)

        # contains paths of archives which contain a nifti image for each subject
        self.images_paths = []
        # contains paths of archives which contain a label (binary brain mask) for
        # each subject
        self.labels_paths = []
        self.images_centers = []  # contains center of each subject: HH, Guys or IOP
        self.images_sets = []  # train and test

        self.subjects = [
            subject
            for subject in os.listdir(self.subjects_dir)
            if os.path.isdir(os.path.join(self.subjects_dir, subject))
        ]
        self.images_centers = [
            _extract_center_name_from_filename(subject) for subject in self.subjects
        ]

        self.demographics = Path(os.path.join(self.subjects_dir, "IXI.xls"))

        for subject in self.subjects:
            patient_id = _get_id_from_filename(subject)
            self.images_sets.append(self.metadata.loc[patient_id, "Split"])
            subject_dir = os.path.join(self.subjects_dir, subject)
            image_path = Path(os.path.join(subject_dir, "T1"))
            label_path = Path(os.path.join(subject_dir, "label"))
            self.images_paths.extend(image_path.glob("*.nii.gz"))
            self.labels_paths.extend(label_path.glob("*.nii.gz"))

        self.filenames = [filename.name for filename in self.images_paths]
        self.subject_ids = tuple(map(_get_id_from_filename, self.filenames))

    @property
    def zip_file(self) -> ZipFile:
        zf = self.root_folder.joinpath(self.parent_folder + ".zip")
        return ZipFile(zf)

    def _validate_center(self) -> None:
        """
        Asserts permitted image center keys.

        Allowed values are:
            - 0
            - 1
            - 2
            - Guys
            - HH
            - IOP

        Raises
        -------
            AssertionError
                If `center` argument is not contained amongst possible centers.
        """
        centers = list(self.CENTER_LABELS.keys()) + list(self.CENTER_LABELS.values())
        assert self.centers[0] in centers, (
            f"Center {self.centers[0]} "
            "is not compatible with this dataset. "
            f"Existing centers can be named as follow: {centers} "
        )

    def __getitem__(self, item) -> Tuple[Tensor, Dict]:
        patient_id = self.subject_ids[item]
        header_img, img, label, center_name = _load_nifti_image_and_label_by_id(
            zip_file=self.zip_file, patient_id=patient_id, modality=self.modality
        )

        default_transform = Compose(
            [ToTensor(), AddChannel(), Resize(self.common_shape)]
        )

        intensity_transform = Compose([NormalizeIntensity()])

        one_hot_transform = Compose([AsDiscrete(to_onehot=2)])

        img = default_transform(img)
        img = intensity_transform(img)
        label = default_transform(label)
        label = one_hot_transform(label)

        # metadata = {
        #     "IXI_ID": patient_id,
        #     "center": center_name,
        #     "center_label": self.CENTER_LABELS[center_name],
        # }

        if self.transform:
            img = self.transform(img)
        return img.to(torch.float32), label

    def __len__(self) -> int:
        return len(self.images_paths)


class FedIXITiny(IXITinyRaw):
    """
    Federated class for T1 images in IXI Tiny Dataset

    Parameters
    ----------
    transform:
        PyTorch Transform to process the data or augment it.
    center: int, optional
        Id of the center (hospital) from which to gather data. Defaults to 0.
    train : bool, optional
        Whether to take the train or test split. Defaults to True (train).
    pooled : bool, optional
        Whether to take all data from the 3 centers into one dataset.
        If True, supersedes center argument. Defaults to False.
    debug : bool, optional
        Default to False.
    data_path: str
        If data_path is given it will ignore the config file and look for the
        dataset directly in data_path. Defaults to None.
    """

    def __init__(
        self,
        transform=None,
        center=0,
        train=True,
        pooled=False,
        debug=False,
        data_path=None,
    ):
        """
        Cf class docstring
        """
        super(FedIXITiny, self).__init__(
            transform=transform, debug=debug, data_path=data_path
        )

        self.modality = "T1"
        self.centers = [center]
        self._validate_center()

        if isinstance(center, int):
            self.centers = [_get_center_name_from_center_id(self.CENTER_LABELS, center)]

        if pooled:
            self.centers = ["Guys", "HH", "IOP"]

        if train:
            self.sets = ["train"]
        else:
            self.sets = ["test"]

        to_select = [
            (self.images_centers[idx] in self.centers)
            and (self.images_sets[idx] in self.sets)
            for idx, _ in enumerate(self.images_centers)
        ]

        self.images_paths = [self.images_paths[i] for i, s in enumerate(to_select) if s]
        self.labels_paths = [self.labels_paths[i] for i, s in enumerate(to_select) if s]
        self.images_centers = [
            self.images_centers[i] for i, s in enumerate(to_select) if s
        ]
        self.images_sets = [self.images_sets[i] for i, s in enumerate(to_select) if s]

        self.filenames = [filename.name for filename in self.images_paths]
        self.subject_ids = tuple(map(_get_id_from_filename, self.filenames))


if __name__ == "__main__":
    a = IXITinyRaw()
    print("IXI Tiny dataset size:", len(a))
    # print('First entry:', a[0])
    a = FedIXITiny()
    print(
        "Data gathered in this federated dataset is from:",
        *a.centers,
        "and",
        *a.sets,
        "set",
    )
    print("Federated dataset size:", len(a))
    print("First entry:", a[0])


__all__ = ["IXITinyRaw", "FedIXITiny"]
