import numpy as np
import torch
from torch.utils.data import Dataset


def split_indices_linear(original_table, dataset_sizes, num_target_centers):
    """Splits linearly the samples from all centers across the new centers,
    in the flattened order

    Parameters
    ----------
    original_table : dict[np.array]
        Dictionary with keys "train", "test", each containing an
        array with 2 lines: original client and indices within them
    dataset_sizes : dict[list]
    num_target_centers : int
        Number of new centers

    Returns
    -------
    dict[np.ndarray]
        Dictionary with the same keys as the input original_table. Each
        entry contains an array with 3 lines: original client,
        indices within them, and
        the assignment to the new client of each sample
    """

    mapping_table = {}

    for split in original_table.keys():
        total_num_samples = original_table[split].shape[-1]
        assignment_new_client = np.zeros(
            total_num_samples, dtype=original_table[split].dtype
        )
        num_samples_per_new_client = total_num_samples // num_target_centers

        current_idx = 0
        for idx_new_client in range(num_target_centers - 1):
            assignment_new_client[
                slice(current_idx, current_idx + num_samples_per_new_client)
            ] = idx_new_client
            current_idx += num_samples_per_new_client
        assignment_new_client[current_idx:] = num_target_centers - 1
        mapping_table[split] = np.concatenate(
            [np.copy(original_table[split]), assignment_new_client.reshape(1, -1)],
            axis=0,
        )

    return mapping_table


def split_indices_dirichlet(
    original_table, dataset_sizes, num_target_centers, dirichlet_param=0.75, seed=42
):
    """Splits the samples from all centers across the new centers,
    based on a Dirichlet sampling which is shared across train and test.

    Parameters
    ----------
    original_table : dict[np.array]
        Dictionary with keys "train", "test", each containing an
        array with 2 lines: original client and indices within them
    num_target_centers : int
        Number of new centers
    dirichlet_param : float, optional
        Number between 0 and 1. The closer to 0, the sharper the distribution
        is. In order to spread the datasets to a larger number of centers,
        prefer values larger around 0.5
    seed : int, optional
        Seed value to use for the random split. Defaults to 42

    Returns
    -------
    dict[np.ndarray]
        Dictionary with the same keys as the input original_table. Each
        entry contains an array with 3 lines: original client,
        indices within them, and
        the assignment to the new client of each sample
    """
    # Compute number of original centers
    orig_centers_indices = np.unique(original_table["train"][0])
    num_orig_centers = orig_centers_indices.size

    # Generate random numbers with seed
    rng = np.random.default_rng(seed)

    # Draw the attribution params for each dataset
    proba_new_centers = rng.dirichlet(
        dirichlet_param * np.ones(num_target_centers), size=(num_orig_centers)
    )
    # Check that the expected number of samples per dataset is large enough
    expected_sizes_new_clients = {
        split: np.dot(dataset_sizes[split], proba_new_centers)
        for split in dataset_sizes.keys()
    }
    for split in expected_sizes_new_clients.keys():
        assert np.all(
            expected_sizes_new_clients[split] > 1
        ), "Not enough samples per center"

    # Compute the mapping table
    mapping_table = {}
    for split in original_table.keys():
        total_num_samples = original_table[split].shape[-1]
        assignment_new_client = np.zeros(
            total_num_samples, dtype=original_table[split].dtype
        )

        # For each original center
        for orig_center_idx in orig_centers_indices:
            # Get the ones corresponding to this value
            idx_samples_in_orig = original_table[split][0, :] == orig_center_idx
            assignment_new_client[idx_samples_in_orig] = np.random.choice(
                np.arange(num_target_centers),
                p=proba_new_centers[orig_center_idx],
                size=(idx_samples_in_orig.sum()),
            )

        mapping_table[split] = np.concatenate(
            [np.copy(original_table[split]), assignment_new_client.reshape(1, -1)],
            axis=0,
        )

    return mapping_table


def get_client_sizes(dataset_class, num_original_centers, debug=False):
    """Gets size of each client in dataset class

    Parameters
    ----------
    dataset_class : Dataset class
        Pytorch class (uninstantiated)
    num_original_centers : int
        Number of original centers
    debug : bool, optional
        Whether to proceed in debug mode, by default False

    Returns
    -------
    dict[List]
        Dictionary with entries "train" and "test", each containing a list
        with the size of each client
    """
    dataset_sizes = {"train": [], "test": []}
    for idx_client in range(num_original_centers):
        for split, is_train in [("train", True), ("test", False)]:
            current_dataset = dataset_class(
                center=idx_client, train=is_train, debug=debug
            )
            dataset_sizes[split].append(len(current_dataset))
    dataset_sizes = {k: np.array(v) for k, v in dataset_sizes.items()}
    return dataset_sizes


def split_dataset(
    dataset_class,
    num_original_centers,
    num_target_centers,
    debug=False,
    seed=42,
    method="dirichlet",
):
    """This function uses the original natural splits found inside the provided
    FLamby dataset class and create a class with num_target_centers by splitting
    the original centers (both train and test) into several fake centers to
    match the target number of centers to create.
    The train-test split is kept the same.
    The resulting class cannot be used in pooled mode.

    Parameters
    ----------
    dataset_class : flamby.datasets
        A Flamby dataset class
    num_original_centers: int
        The number of centers included in this dataset
    num_target_centers: int
        The number of target datasets to create more centers.
        num_target_centers should be greater than num_original_centers.
    debug: bool
        Whether or not to use the dataset in debug mode. Defaults to False.
    seed : int, optional
        Seed to use for the random split (only applicable for Dirichlet).
        Defaults to 42
    method : str, optional
        Method to use to perform the split. Options include "dirichlet"
        and "linear". Defaults to "linear".

    Returns
    -------
    torch.utils.data.Dataset
        The resulting class

    """

    assert (
        num_target_centers >= num_original_centers
    ), "You cannot split a dataset into less centers than it has"
    if num_target_centers == num_original_centers:
        return dataset_class
    else:
        dataset_sizes = get_client_sizes(
            dataset_class, num_original_centers, debug=debug
        )
        total_sizes = {key: np.sum(lengths) for key, lengths in dataset_sizes.items()}

        assert num_target_centers < min(
            total_sizes.values()
        ), f"There are not enough samples to create {num_target_centers}"

        # Compute the original indices table
        original_table = {
            split: np.zeros((2, size), dtype=int) for split, size in total_sizes.items()
        }
        # original_table is a table with 2 columns:
        # original center index
        # index in original center
        for split, client_size_list in dataset_sizes.items():
            _current_idx = 0
            for idx_client_orig, length_client in enumerate(client_size_list):
                original_table[split][
                    0, np.arange(_current_idx, _current_idx + length_client)
                ] = idx_client_orig
                original_table[split][
                    1, np.arange(_current_idx, _current_idx + length_client)
                ] = np.arange(0, length_client)
                _current_idx += length_client

        if method == "linear":
            # Perform a linear split
            mapping_table = split_indices_linear(
                original_table, dataset_sizes, num_target_centers
            )
        elif method == "dirichlet":
            # perf a dirichlet split
            mapping_table = split_indices_dirichlet(
                original_table, dataset_sizes, num_target_centers, seed=seed
            )
        else:
            raise ValueError(f"Unknown split method {method}")

        class SplitDataset(Dataset):
            """This class is basically a dataset_class but can be instantiated
            with num_target_centers different centers that will fetch data from
            the original centers.
            It cannot be used in pooled mode.

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
                self._original_datasets = [
                    dataset_class(i, train, False, X_dtype, y_dtype, debug, data_path)
                    for i in range(num_original_centers)
                ]
                self.center = center
                split_key = "train" if train else "test"

                # Get the sub-array corresponding to the samples of my dataset
                self._mapping_table = mapping_table[split_key][
                    :, mapping_table[split_key][2] == self.center
                ]
                # Make some checks
                assert np.all(self._mapping_table[2, :] == self.center)

            def __len__(self):
                return self._mapping_table.shape[-1]

            def __getitem__(self, idx):
                orig_center_idx = self._mapping_table[0, idx]
                idx_in_orig_center = self._mapping_table[1, idx]

                return self._original_datasets[orig_center_idx][idx_in_orig_center]

        return SplitDataset


if __name__ == "__main__":
    from torch.utils.data import DataLoader as dl

    from flamby.datasets.fed_heart_disease import NUM_CLIENTS, FedHeartDisease

    ntarget = 20
    BATCH_SIZE = 2
    # original dataset
    train_dls = [
        dl(FedHeartDisease(center=i, train=True), batch_size=BATCH_SIZE, shuffle=True)
        for i in range(NUM_CLIENTS)
    ]
    test_dls = [
        dl(FedHeartDisease(center=i, train=False), batch_size=BATCH_SIZE, shuffle=False)
        for i in range(NUM_CLIENTS)
    ]
    print("Finished loading original datasets")
    # new dataset
    FedHeartDiseaseSplit20 = split_dataset(FedHeartDisease, NUM_CLIENTS, ntarget)

    new_train_dls = [
        dl(
            FedHeartDiseaseSplit20(center=i, train=True),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        for i in range(ntarget)
    ]
    new_test_dls = [
        dl(
            FedHeartDiseaseSplit20(center=i, train=False),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        for i in range(ntarget)
    ]
    print("created new datasets")
    assert sum([len(t.dataset) for t in train_dls]) == sum(
        [len(t.dataset) for t in new_train_dls]
    )
    assert sum([len(t.dataset) for t in test_dls]) == sum(
        [len(t.dataset) for t in new_test_dls]
    )

    # Check that one can load everything:
    for i in range(ntarget):
        for X, y in new_train_dls[i]:
            pass
        for X, y in new_test_dls[i]:
            pass
