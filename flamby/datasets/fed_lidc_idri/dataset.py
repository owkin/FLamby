import os
from pathlib import Path

import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import Dataset

import flamby.datasets.fed_lidc_idri
from flamby.datasets.fed_lidc_idri.data_utils import (
    ClipNorm,
    Sampler,
    resize_by_crop_or_pad,
)
from flamby.utils import check_dataset_from_config


class LidcIdriRaw(Dataset):
    """
    Pytorch dataset containing all the features, labels and
    metadata for LIDC-IDRI without any discrimination.

    Attributes
    ----------
    ctscans_dir : str
        The directory where ctscans are located.
    metadata: pd.DataFrame
        The ground truth dataframe for metadata such as centers
    features_paths: list[str]
        The list with the paths towards all features.
    masks_paths: list[int]
        The list with the paths towards segmentation masks
    features_centers: list[int]
        The list for all centers for all features
    features_sets: list[str]
        The list for all sets (train/test) for all features
    X_dtype: torch.dtype
        The dtype of the X features output
    y_dtype: torch.dtype
        The dtype of the y label output
    debug : bool
        whether the dataset was processed in debug mode (first 10 files)
    transform : torch.torchvision.Transform or None
        Transformation to perform on data.
    out_shape : Tuple or None
        The desired output shape (If None, no reshaping)
    sampler: Sampler object
        algorithm to sample patches

    Parameters
    ----------
    X_dtype : torch.dtype, optional
        Dtype for inputs `X`. Defaults to `torch.float32`.
    y_dtype : torch.dtype, optional
        Dtype for labels `y`. Defaults to `torch.int64`.
    sampler : flamby.datasets.fed_lidc_idri.data_utils.Sampler
        Patch sampling method.
    transform : torch.torchvision.Transform or None, optional.
        Transformation to perform on each data point. Default: ClipNorm.
    out_shape : Tuple or None, optional
        The desired output shape. If None, no padding or cropping is performed.
        Default is (384, 384, 384).
    debug : bool, optional
        Whether the dataset was downloaded in debug mode. Defaults to false.
    data_path: str
        If data_path is given it will ignore the config file and look for the
        dataset directly in data_path. Defaults to None.
    """

    def __init__(
        self,
        X_dtype=torch.float32,
        y_dtype=torch.int64,
        out_shape=(384, 384, 384),
        sampler=Sampler(),
        transform=ClipNorm(),
        debug=False,
        data_path=None,
    ):
        """
        Cf class docstring
        """
        self.metadata = pd.read_csv(
            Path(os.path.dirname(flamby.datasets.fed_lidc_idri.__file__))
            / Path("metadata")
            / Path("metadata.csv")
        )
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.out_shape = out_shape
        self.transform = transform
        self.features_paths = []
        self.masks_paths = []
        self.features_centers = []
        self.features_sets = []
        self.debug = debug
        self.sampler = sampler
        if data_path is None:
            config_dict = check_dataset_from_config(
                dataset_name="fed_lidc_idri", debug=debug
            )
            self.ctscans_dir = Path(config_dict["dataset_path"])
        else:
            if not (os.path.exists(data_path)):
                raise ValueError(f"The string {data_path} is not a valid path.")
            self.ctscans_dir = Path(data_path)

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
        # Load nifti files, and convert them to torch
        X = nib.load(self.features_paths[idx])
        y = nib.load(self.masks_paths[idx])
        X = torch.from_numpy(X.get_fdata()).to(self.X_dtype)
        y = torch.from_numpy(y.get_fdata()).to(self.y_dtype)
        # CT scans have different sizes. Crop or pad to desired common shape.
        X = resize_by_crop_or_pad(X, self.out_shape)
        y = resize_by_crop_or_pad(y, self.out_shape)
        # Apply optional additional transforms, such as normalization
        if self.transform is not None:
            X = self.transform(X)
        # Sample and return patches
        return self.sampler(X, y)


class FedLidcIdri(LidcIdriRaw):
    """
    Pytorch dataset containing for each center the features and associated labels
    for LIDC-IDRI federated classification.

    Parameters
    ----------
    X_dtype : torch.dtype, optional
        Dtype for inputs `X`. Defaults to `torch.float32`.
    y_dtype : torch.dtype, optional
        Dtype for labels `y`. Defaults to `torch.int64`.
    out_shape : Tuple or None, optional
        The desired output shape. If None, no padding or cropping is performed.
        Default is (384, 384, 384).
    sampler : flamby.datasets.fed_lidc_idri.data_utils.Sampler
        Patch sampling method.
    transform : torch.torchvision.Transform or None, optional.
        Transformation to perform on each data point.
    center : int, optional
        Id of the center from which to gather data. Defaults to 0.
    train : bool, optional
        Whether to take the train or test split. Defaults to True (train).
    pooled : bool, optional
        Whether to take all data from the 2 centers into one dataset.
        If True, supersedes center argument. Defaults to False.
    debug : bool, optional
        Whether the dataset was downloaded in debug mode. Defaults to false.
    data_path: str
        If data_path is given it will ignore the config file and look for the
        dataset directly in data_path. Defaults to None.
    """

    def __init__(
        self,
        X_dtype=torch.float32,
        y_dtype=torch.int64,
        out_shape=(384, 384, 384),
        sampler=Sampler(),
        transform=ClipNorm(),
        center=0,
        train=True,
        pooled=False,
        debug=False,
        data_path=None,
    ):
        """
        Cf class docstring
        """

        super().__init__(
            X_dtype=X_dtype,
            y_dtype=y_dtype,
            out_shape=out_shape,
            sampler=sampler,
            transform=transform,
            debug=debug,
            data_path=data_path,
        )

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

        if not train:
            self.sampler = Sampler(algo="all")


def collate_fn(dataset_elements_list):
    """Helper function to correctly batch samples from
    a LidcIdriDataset, taking patch sampling into account.
    Parameters
    ----------
    dataset_elements_list : List[(torch.Tensor, torch.Tensor)]
        List of batches of samples from ct scans and masks.
        The list has length B, tensors have shape (S, D, W, H).
    Returns
    -------
    Tuple(torch.Tensor, torch.Tensor)
        X, y two torch tensors of size (B * S, 1, D, W, H)
    """
    X, y = zip(*dataset_elements_list)
    X, y = torch.cat(X), torch.cat(y)
    # Check that images and mask have a channel dimension
    if X.ndim == 5:
        return X, y
    else:
        return X.unsqueeze(1), y.unsqueeze(1)
