import os
from pathlib import Path

import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import Dataset

import flamby.datasets.fed_lidc_idri
from flamby.datasets.fed_lidc_idri import METADATA_DICT

dic = METADATA_DICT


class LidcIdriRaw(Dataset):
    """
    Pytorch dataset containing all the features, labels and
    metadata for LIDC-IDRI without any discrimination.

    Attributes
    ----------
    """

    def __init__(self, ctscans_dir=".", X_dtype=torch.float32, y_dtype=torch.float32):
        """
        Parameters
        ----------
        """
        # Add ctscans_dir as an argument
        # TODO : check with Jean if this is ok
        self.ctscans_dir = Path(ctscans_dir)
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

        # TODO : in the Luna16 challenge, slides that are too thick were excluded
        # (~130 slides)
        # Should we do the same?

        for ctscan in self.ctscans_dir.rglob("*patient.nii.gz"):
            ctscan_name = os.path.basename(os.path.dirname(ctscan))
            mask_path = os.path.join(os.path.dirname(ctscan), "mask_consensus.nii.gz")

            center_from_metadata = self.metadata[
                self.metadata.SeriesInstanceUID == ctscan_name
            ].Manufacturer.item()

            # TODO: replace with true train/test split
            if True:
                self.features_sets.append("train")
            else:
                self.features_sets.append("test")

            self.features_paths.append(ctscan)
            # TODO: do we need both mask and mask_consensus?
            self.masks_paths.append(mask_path)
            self.features_centers.append(center_from_metadata)

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
        ctscans_dir=".",
        X_dtype=torch.float32,
        y_dtype=torch.int64,
        center=0,
        train=True,
        pooled=False,
    ):

        super().__init__(ctscans_dir=ctscans_dir, X_dtype=X_dtype, y_dtype=y_dtype)

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
