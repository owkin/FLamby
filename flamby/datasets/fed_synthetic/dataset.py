import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from flamby.datasets.fed_synthetic.synthetic_generator import generate_synthetic_dataset


class SyntheticRaw(Dataset):
    """Pytorch dataset containing all the features, labels and
    metadata for the Synthetic dataset.
    Attributes
    ----------
    data_dir : str, Where data files are located
    data_paths: list[str], The list with the path towards all features.
    features_labels: list[int], The list with all classification labels for all features
    features_centers: list[int], The list for all centers for all features
    features_sets: list[str], The list for all sets (train/test) for all features
    X_dtype: torch.dtype, The dtype of the X features output
    y_dtype: torch.dtype, The dtype of the y label output
    debug: bool, Whether or not we use the dataset with only part of the features
    """

    def __init__(
        self,
        n_centers=5,
        n_samples=[1000, 10, 100, 500],
        n_features=10,
        seed=42,
        X_dtype=torch.float32,
        y_dtype=torch.float32,
        debug=False,
    ):
        """See description above
        Parameters
        ----------
        n_centers : int, optional
            Number of centers in the dataset.
        n_samples : int or list, optional
            Number of records in each center.
        n_features : int, optional
            Number of features in the dataset.
        seed : int, optional
            Seed for the random data generation.
        X_dtype : torch.dtype, optional
            Dtype for inputs `X`. Defaults to `torch.float32`.
        y_dtype : torch.dtype, optional
            Dtype for labels `y`. Defaults to `torch.int64`.
        debug : bool, optional,
            Whether or not to use only the part of the dataset downloaded in
            debug mode. Defaults to False.
        """

        # dict, config_file = create_config(output_folder, debug, "fed_heart_disease")

        # dict = check_dataset_from_config("fed_synthetic", debug)
        # self.data_dir = Path(dict["dataset_path"])

        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.debug = debug

        self.n_centers = n_centers
        self.n_samples = n_samples
        self.n_features = n_features

        self.features = pd.DataFrame()
        self.labels = pd.DataFrame()
        self.centers = []
        self.sets = []

        self.train_fraction = 0.66

        full_df, indices = generate_synthetic_dataset(n_centers, n_samples, n_features)

        for i, center_indices in enumerate(indices):

            df = full_df.iloc[center_indices]

            center_X = df.iloc[:, :-1]
            center_y = df.iloc[:, -1]

            self.features = pd.concat((self.features, center_X), ignore_index=True)
            self.labels = pd.concat((self.labels, center_y), ignore_index=True)

            self.centers += [i] * center_X.shape[0]

            # proposed modification to introduce shuffling before splitting the center
            nb = int(center_X.shape[0])
            current_labels = center_y.where(center_y == 0, 1, inplace=False)
            levels = np.unique(current_labels)
            if (len(np.unique(current_labels)) > 1) and all(
                [(current_labels == lev).sum() > 2 for lev in levels]
            ):
                stratify = current_labels
            else:
                stratify = None
            indices_train, indices_test = train_test_split(
                np.arange(nb),
                test_size=1.0 - self.train_fraction,
                train_size=self.train_fraction,
                random_state=43,
                shuffle=True,
                stratify=stratify,
            )

            for i in np.arange(nb):
                if i in indices_test:
                    self.sets.append("test")
                else:
                    self.sets.append("train")

        # keep 0 (no disease) and put 1 for all other values (disease)
        self.labels.where(self.labels == 0, 1, inplace=True)
        self.labels = torch.from_numpy(self.labels.values).to(self.X_dtype)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        assert idx < len(self.features), "Index out of range."

        X = self.features[idx]
        y = self.labels[idx]

        return X, y


class FedSynthetic(SyntheticRaw):
    """
    Pytorch dataset containing for each center the features and associated labels
    for Synthetic federated classification.
    One can instantiate this dataset with train or test data coming from either
    of the 4 centers it was created from or all data pooled.
    The train/test split are arbitrarily fixed.
    """

    def __init__(
        self,
        n_centers=5,
        n_samples=[1000, 10, 100, 500],
        n_features=10,
        seed=42,
        center=0,
        train=True,
        pooled=False,
        X_dtype=torch.float32,
        y_dtype=torch.float32,
        debug=False,
    ):
        """Instantiate the dataset
        Parameters
        n_centers : int, optional
            Number of centers in the dataset.
        n_samples : int or list, optional
            Number of records in each center.
        n_features : int, optional
            Number of features in the dataset.
        seed : int, optional
            Seed for the random data generation.
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
        """

        super().__init__(X_dtype=X_dtype, y_dtype=y_dtype, debug=debug)
        assert center in np.arange(self.n_centers)

        self.chosen_centers = [center]
        if pooled:
            self.chosen_centers = np.arange(self.n_centers)

        if train:
            self.chosen_sets = ["train"]
        else:
            self.chosen_sets = ["test"]

        to_select = [
            (self.sets[idx] in self.chosen_sets)
            and (self.centers[idx] in self.chosen_centers)
            for idx, _ in enumerate(self.features.index)
        ]

        self.features = [fp for idx, fp in enumerate(self.features) if to_select[idx]]
        self.sets = [fp for idx, fp in enumerate(self.sets) if to_select[idx]]
        self.labels = [fp for idx, fp in enumerate(self.labels) if to_select[idx]]
        self.centers = [fp for idx, fp in enumerate(self.centers) if to_select[idx]]
