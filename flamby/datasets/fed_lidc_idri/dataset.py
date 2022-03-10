import os
from pathlib import Path

import pandas as pd
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

    def __init__(self, ctscans_dir="."):
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
        self.features_paths = []
        self.masks_paths = []
        self.features_centers = []
        self.features_sets = []

        for ctscan in self.ctscans_dir.rglob("*patient.nii.gz"):
            ctscan_name = os.path.basename(os.path.dirname(ctscan))
            mask_path = os.path.join(os.path.dirname(ctscan), "mask.nii.gz")

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
        # get feature + label/mask
        pass


class FedLidcIdri(LidcIdriRaw):
    """
    Pytorch dataset containing for each center the features and associated labels
    for LIDC-IDRI federated classification.
    """

    def __init__(self, center=0, train=True, pooled=False):
        super().__init__()
        pass
