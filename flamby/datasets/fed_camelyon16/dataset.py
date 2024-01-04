import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import flamby.datasets.fed_camelyon16
from flamby.utils import check_dataset_from_config


class Camelyon16Raw(Dataset):
    """Pytorch dataset containing all the features, labels and
    metadata for Camelyon16 WS without any discrimination.

    Parameters
    ----------
    X_dtype : torch.dtype, optional
        Dtype for inputs `X`. Defaults to `torch.float32`.
    y_dtype : torch.dtype, optional
        Dtype for labels `y`. Defaults to `torch.int64`.
    debug : bool, optional,
        Whether or not to use only the part of the dataset downloaded in
        debug mode. Defaults to False.
    data_path: str
        If data_path is given it will ignore the config file and look for the
        dataset directly in data_path. Defaults to None.

    Attributes
    ----------
    tiles_dir : str
        Where all features are located
    labels: pd.DataFrame
        The ground truth DataFrame for labels
    metadata: pd.DataFrame
        The ground truth dataframe for metadata such as centers
    features_paths: list[str]
        The list with the path towards all features.
    features_labels: list[int]
        The list with all classification labels for all features
    features_centers: list[int]
        The list for all centers for all features
    features_sets: list[str]
        The list for all sets (train/test) for all features
    X_dtype: torch.dtype
        The dtype of the X features output
    y_dtype: torch.dtype
        The dtype of the y label output
    debug: bool
        Whether or not we use the dataset with only part of the features
    perms: dict
        The dictionary of all generated permutations for each slide.
    """

    def __init__(
        self, X_dtype=torch.float32, y_dtype=torch.float32, debug=False, data_path=None
    ):
        """See description above"""
        if data_path is None:
            dict = check_dataset_from_config("fed_camelyon16", debug)
            self.tiles_dir = Path(dict["dataset_path"])
        else:
            if not (os.path.exists(data_path)):
                raise ValueError(f"The string {data_path} is not a valid path.")
            self.tiles_dir = Path(data_path)

        path_to_labels_file = str(
            Path(
                os.path.dirname(flamby.datasets.fed_camelyon16.__file__)
                / Path("labels.csv")
            )
        )
        self.labels = pd.read_csv(path_to_labels_file, index_col="filenames")
        self.metadata = pd.read_csv(
            Path(os.path.dirname(flamby.datasets.fed_camelyon16.__file__))
            / Path("metadata")
            / Path("metadata.csv")
        )
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.debug = debug
        self.features_paths = []
        self.features_labels = []
        self.features_centers = []
        self.features_sets = []
        self.perms = {}
        # We need this list to be sorted for reproducibility but shuffled to
        # avoid weirdness
        # filter out normal_086 and test_049 slides since they have been
        # removed from the Camelyon16 dataset
        npys_list = [
            e
            for e in sorted(self.tiles_dir.glob("*.npy"))
            if e.name.lower() not in ("normal_086.tif.npy", "test_049.tif.npy")
        ]
        random.seed(0)
        random.shuffle(npys_list)
        for slide in npys_list:
            slide_name = os.path.basename(slide).split(".")[0].lower()
            slide_id = int(slide_name.split("_")[1])
            label_from_metadata = int(
                self.metadata.loc[
                    [
                        e.split(".")[0] == slide_name
                        for e in self.metadata["slide_name"].tolist()
                    ],
                    "label",
                ].item()
            )
            center_from_metadata = int(
                self.metadata.loc[
                    [
                        e.split(".")[0] == slide_name
                        for e in self.metadata["slide_name"].tolist()
                    ],
                    "hospital_corrected",
                ].item()
            )
            label_from_data = int(self.labels.loc[slide.name.lower()].tumor)

            if "test" not in str(slide).lower():
                if slide_name.startswith("normal"):
                    # Normal slide
                    if slide_id > 100:
                        center_label = 1

                    else:
                        center_label = 0
                    label_from_slide_name = 0  # Normal slide
                elif slide_name.startswith("tumor"):
                    # Tumor slide
                    if slide_id > 70:
                        center_label = 1
                    else:
                        center_label = 0
                    label_from_slide_name = 1  # Tumor slide
                self.features_sets.append("train")
                assert label_from_slide_name == label_from_data, "This shouldn't happen"
                assert center_label == center_from_metadata, "This shouldn't happen"

            else:
                self.features_sets.append("test")

            assert label_from_metadata == label_from_data
            self.features_paths.append(slide)
            self.features_labels.append(label_from_data)
            self.features_centers.append(center_from_metadata)
        if len(self.features_paths) < len(self.labels.index):
            if not (self.debug):
                logging.warning(
                    f"You have {len(self.features_paths)} features found in"
                    f" {str(self.tiles_dir)} instead of {len(self.labels.index)} (full"
                    " Camelyon16 dataset), please go back to the installation"
                    " instructions."
                )
            else:
                print(
                    "Warning you are operating on a reduced dataset in  DEBUG mode with"
                    " in total {len(self.features_paths)}/{len(self.labels.index)}"
                    " features."
                )

    def __len__(self):
        return len(self.features_paths)

    def __getitem__(self, idx):
        start = 0
        X = np.load(self.features_paths[idx])[:, start:]
        X = torch.from_numpy(X).to(self.X_dtype)
        y = torch.from_numpy(np.asarray(self.features_labels[idx])).to(self.y_dtype)
        if idx not in self.perms:
            self.perms[idx] = np.random.default_rng(42).permutation(X.shape[0])

        return X, y, self.perms[idx]


class FedCamelyon16(Camelyon16Raw):
    """
    Pytorch dataset containing for each center the features and associated labels
    for Camelyon16 federated classification.
    One can instantiate this dataset with train or test data coming from either
    of the 2 centers it was created from or all data pooled.
    The train/test split corresponds to the one from the Challenge.

    Parameters
    ----------
    center : int, optional
        Default to 0.
    train : bool, optional
        Default to True
    pooled : bool, optional
        Whether to take all data from the 2 centers into one dataset, by
        default False
    X_dtype : torch.dtype, optional
        Dtype for inputs `X`. Defaults to `torch.float32`.
    y_dtype : torch.dtype, optional
        Dtype for labels `y`. Defaults to `torch.int64`.
    debug : bool, optional,
        Whether or not to use only the part of the dataset downloaded in
        debug mode. Defaults to False.
    data_path: str
        If data_path is given it will ignore the config file and look for the
        dataset directly in data_path. Defaults to None.

    """

    def __init__(
        self,
        center: int = 0,
        train: bool = True,
        pooled: bool = False,
        X_dtype: torch.dtype = torch.float32,
        y_dtype: torch.dtype = torch.float32,
        debug: bool = False,
        data_path: str = None,
    ):
        """
        Cf class docstring
        """
        super().__init__(
            X_dtype=X_dtype, y_dtype=y_dtype, debug=debug, data_path=data_path
        )
        assert center in [0, 1]
        self.centers = [center]
        if pooled:
            self.centers = [0, 1]
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
        self.features_labels = [
            fp for idx, fp in enumerate(self.features_labels) if to_select[idx]
        ]
        self.features_centers = [
            fp for idx, fp in enumerate(self.features_centers) if to_select[idx]
        ]


def collate_fn(dataset_elements_list, max_tiles=10000):
    """Helper function to correctly batch samples from
    a Camelyon16Dataset accomodating for the uneven number of tiles per slide.

    Parameters
    ----------
    dataset_elements_list : List[torch.Tensor]
        A list of torch tensors of dimensions [n, m] with uneven distribution of ns.
    max_tiles : int, optional
        The nummber of tiles max by Tensor, by default 10000

    Returns
    -------
    Tuple(torch.Tensor, torch.Tensor)
        X, y two torch tensors of size (len(dataset_elements_list), max_tiles, m) and
        (len(dataset_elements_list),)
    """
    n = len(dataset_elements_list)
    X0, y0, _ = dataset_elements_list[0]
    feature_dim = X0.size(1)
    X_dtype = X0.dtype
    y_dtype = y0.dtype
    X = torch.zeros((n, max_tiles, feature_dim), dtype=X_dtype)
    y = torch.empty((n, 1), dtype=y_dtype)

    for i in range(n):
        X_current, y_current, perm = dataset_elements_list[i]
        ntiles_min = min(max_tiles, X_current.shape[0])
        X[i, :ntiles_min, :] = X_current[perm[:ntiles_min], :]
        y[i] = y_current
    return X, y
