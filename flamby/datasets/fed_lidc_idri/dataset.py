import os
from pathlib import Path

import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import Dataset

import flamby.datasets.fed_lidc_idri
from flamby.datasets.fed_lidc_idri import METADATA_DICT
from flamby.utils import check_dataset_from_config

dic = METADATA_DICT


class LidcIdriRaw(Dataset):
    """
    Pytorch dataset containing all the features, labels and
    metadata for LIDC-IDRI without any discrimination.

    Attributes
    ----------
    ctscans_dir : str, The directory where ctscans are located.
    metadata: pd.DataFrame, The ground truth dataframe for metadata such as centers
    features_paths: list[str], The list with the paths towards all features.
    masks_paths: list[int], The list with the paths towards segmentation masks
    features_centers: list[int], The list for all centers for all features
    features_sets: list[str], The list for all sets (train/test) for all features
    X_dtype: torch.dtype, The dtype of the X features output
    y_dtype: torch.dtype, The dtype of the y label output
    debug : bool, whether the dataset was processed in debug mode (first 10 files)
    """

    def __init__(self, X_dtype=torch.float32, y_dtype=torch.int64, debug=False):
        """
        See description above
        Parameters
        ----------
        X_dtype : torch.dtype, optional
          Dtype for inputs `X`. Defaults to `torch.float32`.
        y_dtype : torch.dtype, optional
          Dtype for labels `y`. Defaults to `torch.int64`.
        debug : bool, optional
            Whether the dataset was downloaded in debug mode. Defaults to false.
        """
        self.metadata = pd.read_csv(
            Path(os.path.dirname(flamby.datasets.fed_lidc_idri.__file__))
            / Path("metadata")
            / Path("metadata.csv")
        )
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.features_paths = []
        self.masks_paths = []
        self.features_centers = []
        self.features_sets = []
        self.debug = debug

        config_dict = check_dataset_from_config(debug)
        self.ctscans_dir = Path(config_dict["dataset_path"])

        for ctscan in self.ctscans_dir.rglob("*patient.nii.gz"):
            ctscan_name = os.path.basename(os.path.dirname(ctscan))
            mask_path = os.path.join(os.path.dirname(ctscan), "mask_consensus.nii.gz")

            center_from_metadata = self.metadata[
                self.metadata.SeriesInstanceUID == ctscan_name
            ].Manufacturer.item()

            split_from_metadata = self.metadata[
                self.metadata.SeriesInstanceUID == ctscan_name
            ].Split.item()

            self.features_paths.append(ctscan)
            self.masks_paths.append(mask_path)
            self.features_centers.append(center_from_metadata)
            self.features_sets.append(split_from_metadata)

    def __len__(self):
        return len(self.features_paths)

    def __getitem__(self, idx):
        X = nib.load(self.features_paths[idx])
        X = torch.from_numpy(X.get_fdata()).to(self.X_dtype)
        y = nib.load(self.masks_paths[idx])
        y = torch.from_numpy(y.get_fdata()).to(self.y_dtype)
        return X, y


class FedLidcIdri(LidcIdriRaw):
    """
    Pytorch dataset containing for each center the features and associated labels
    for LIDC-IDRI federated classification.
    """

    def __init__(
        self,
        X_dtype=torch.float32,
        y_dtype=torch.int64,
        center=0,
        train=True,
        pooled=False,
        debug=False,
    ):
        """
        Instantiate the dataset
        Parameters
        ----------
        X_dtype : torch.dtype, optional
            Dtype for inputs `X`. Defaults to `torch.float32`.
        y_dtype : torch.dtype, optional
            Dtype for labels `y`. Defaults to `torch.int64`.
        center : int, optional
            Id of the center from which to gather data. Defaults to 0.
        train : bool, optional
            Whether to take the train or test split. Defaults to True (train).
        pooled : bool, optional
            Whether to take all data from the 2 centers into one dataset.
            If True, supersedes center argument. Defaults to False.
        debug : bool, optional
            Whether the dataset was downloaded in debug mode. Defaults to false.
        """

        super().__init__(X_dtype=X_dtype, y_dtype=y_dtype, debug=debug)

        assert center in [0, 1, 2, 3]
        self.centers = [center]
        if pooled:
            self.centers = [0, 1, 2, 3]
        if train:
            self.sets = ["train"]
        else:
            self.sets = ["test"]

        to_select = [
            (self.features_sets[idx] in self.sets)
            and (self.features_centers[idx] in self.centers)
            for idx, _ in enumerate(self.features_centers)
        ]
        self.features_paths = [
            fp for idx, fp in enumerate(self.features_paths) if to_select[idx]
        ]
        self.features_sets = [
            fp for idx, fp in enumerate(self.features_sets) if to_select[idx]
        ]
        self.masks_paths = [
            fp for idx, fp in enumerate(self.masks_paths) if to_select[idx]
        ]
        self.features_centers = [
            fp for idx, fp in enumerate(self.features_centers) if to_select[idx]
        ]
