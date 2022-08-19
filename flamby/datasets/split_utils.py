import torch
from torch.utils.data import Dataset

# def split_indices():


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
    return dataset_sizes


def split_dataset(dataset_class, num_original_centers, num_target_centers, debug=False):
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
        total_sizes = {key: sum(lengths) for key, lengths in dataset_sizes.items()}

        assert num_target_centers < min(
            total_sizes.values()
        ), f"There are not enough samples to create {num_target_centers}"

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
                self.datasets = [
                    dataset_class(i, train, False, X_dtype, y_dtype, debug, data_path)
                    for i in range(num_original_centers)
                ]
                self.center = center
                split_key = "train" if train else "test"
                total_size = sum(dataset_sizes[split_key])
                # We flatten all the original datasets one after the other
                self.ids_datasets = []
                self.indices_in_orig_center = []
                for idx_original_client in range(len(dataset_sizes[split_key])):
                    for idx_sample in range(
                        dataset_sizes[split_key][idx_original_client]
                    ):
                        self.ids_datasets.append(idx_original_client)
                        self.indices_in_orig_center.append(idx_sample)

                # every dataset has size
                self.mean_dataset_size = total_size // num_target_centers
                # except the last one in which we put the rest
                last_center_size = self.mean_dataset_size + (
                    total_size - self.mean_dataset_size * num_target_centers
                )
                self.size = (
                    self.mean_dataset_size
                    if center != (num_target_centers - 1)
                    else last_center_size
                )

            def __len__(self):
                return self.size

            def __getitem__(self, idx):

                idx_flattened = idx + self.center * self.mean_dataset_size
                orig_center_index = self.ids_datasets[idx_flattened]
                idx_in_orig_center = self.indices_in_orig_center[idx_flattened]

                return self.datasets[orig_center_index][idx_in_orig_center]

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
    print("Finished loading origiinal datasets")
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

    for i in range(ntarget):
        for X, y in new_train_dls[i]:
            pass
        for X, y in new_test_dls[i]:
            pass
