import os
from pathlib import Path

import pandas as pd
import torch


class TcgaBrcaRaw(torch.utils.data.Dataset):
    """Pytorch dataset containing all the clinical features and (event, time)
    information for TCGA-BRCA survival analysis.
    Attributes
    ----------
    X_dtype: torch.dtype, the dtype of the X features output
    y_dtype: torch.dtype, the dtype of the (E, T) output
    train_test: str, equals "train" or "test", characterizes if the dataset is
    used for training or for testing, default "train"
    dic: dictionary containing the paths to the data and the train_test_split file
    data: pandas dataframe containing the data for the patients in the training
    or the test set according to the variable train_test
    __getitem__: returns a tuple, first element is a torch tensor of dimension
    (39,) for the covariates, second element is a torch tensor of dimension (2,)
    for E, T
    """

    def __init__(self, train=True, X_dtype=torch.float32, y_dtype=torch.float32):

        input_path = Path(os.path.realpath(__file__)).parent.resolve()
        self.dic = {
            "input_preprocessed": os.path.join(input_path, "brca.csv"),
            "train_test_split": os.path.join(
                input_path, "dataset_creation_scripts/train_test_split.csv"
            ),
        }
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.train_test = "train" if train else "test"
        df = pd.read_csv(self.dic["train_test_split"])
        pids = df.query("fold == '" + self.train_test + "' ").reset_index(drop=True)
        pid_list = list(pids["pid"])
        df2 = pd.read_csv(self.dic["input_preprocessed"])
        self.data = df2[df2["pid"].isin(pid_list)]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data.iloc[idx, 1:40]
        y = self.data.iloc[idx, 40:42]
        return (
            torch.tensor(x, dtype=self.X_dtype),
            torch.tensor(y, dtype=self.y_dtype),
        )


class FedTcgaBrca(TcgaBrcaRaw):
    """
    Pytorch dataset containing all the clinical features and (event, time)
    information for TCGA-BRCA survival analysis.
    One can instantiate this dataset with train or test data coming from either of
    the 6 regions or all regions pooled.
    The train/test split is static and given in the train_test_split file.
    Attributes
    ----------
    pooled: boolean, characterizes if the dataset is pooled or not
    center: int, between 0 and 5, designates the region in the case of pooled==False
    data: pandas dataframe containing the data for the patients in the training
    or the test set (according to the variable train) in a specific region
    (chosen thanks to the variable center if pooled==False) or across all regions
    (if pooled==True)
    """

    def __init__(
        self,
        center=0,
        train=True,
        pooled=False,
        X_dtype=torch.float32,
        y_dtype=torch.float32,
    ):

        super().__init__(train, X_dtype=X_dtype, y_dtype=y_dtype)

        self.center = center
        self.pooled = pooled
        key = self.train_test + "_" + str(self.center)
        if not self.pooled:
            assert center in range(6)
            df = pd.read_csv(self.dic["train_test_split"])
            pids = df.query("fold2 == '" + key + "' ").reset_index(drop=True)
            pid_list = list(pids["pid"])
            df2 = pd.read_csv(self.dic["input_preprocessed"])
            self.data = df2[df2["pid"].isin(pid_list)]


if __name__ == "__main__":

    mydataset = FedTcgaBrca(train=True, pooled=True)
    print(f"The dataset has {len(mydataset)} elements")
    print("Example of dataset record: ", mydataset[0])

    mydataset = FedTcgaBrca(center=5, train=False, pooled=False)
    print(f"The dataset has {len(mydataset)} elements")
    for i in range(5):
        print(f"X {i} ", mydataset[i][0], mydataset[i][0].shape)
        print(f"(E,T) {i} ", mydataset[i][1])
