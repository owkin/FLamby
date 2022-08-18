import torch
from torch.utils.data import Dataset


def split_dataset(dataset_class, num_original_centers, num_target_centers, debug=False):
    """This function uses the original natural splits found inside the provided
    FLamby dataset class and create a class with num_target_centers by splitting
    the original centers (both train and test) into several fake centers to
    match the target number of centers to create. The train-test split is kept
    the same.
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
        datasets = []
        for i in range(num_original_centers):
            current_dataset_train = dataset_class(center=i, train=True, debug=debug)
            current_dataset_test = dataset_class(center=i, train=False, debug=debug)
            datasets.append(
                {
                    "train": {
                        "dataset": current_dataset_train,
                        "size": len(current_dataset_train),
                    },
                    "test": {
                        "dataset": current_dataset_test,
                        "size": len(current_dataset_test),
                    },
                }
            )

        total_size_train = sum([d["train"]["size"] for d in datasets])
        total_size_test = sum([d["test"]["size"] for d in datasets])
        assert num_target_centers < min(
            total_size_test, total_size_train
        ), f"There is not enough samples to create {num_target_centers}"

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
                datasets_sizes = [d[split_key]["size"] for d in datasets]
                total_size = sum(datasets_sizes)
                # We flatten all the original datasets one after the other
                self.ids_datasets = []
                self.indices_in_orig_center = []
                for didx in range(len(datasets_sizes)):
                    for i in range(datasets_sizes[didx]):
                        self.ids_datasets.append(didx)
                        self.indices_in_orig_center.append(i)

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
