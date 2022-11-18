import os
from pathlib import Path

import pandas as pd
import torch

from flamby.utils import accept_license


class TcgaBrcaRaw(torch.utils.data.Dataset):
    """Pytorch dataset containing all the clinical features and (event, time)
    information for TCGA-BRCA survival analysis.
    Attributes
    ----------
    X_dtype: torch.dtyp
        the dtype of the X features output
    y_dtype: torch.dtype,
        the dtype of the (E, T) output
    dic:
        dictionary containing the paths to the data and the train_test_split file
    data:
        pandas dataframe containing the data for the all the patients
    __getitem__:
        returns a tuple, first element is a torch tensor of dimension
        (39,) for the covariates, second element is a torch tensor of dimension (2,)
        for E, T
    """

    def __init__(self, X_dtype=torch.float32, y_dtype=torch.float32):
        accept_license(
            "https://gdc.cancer.gov/access-data/data-access-processes-and-tools",
            "fed_tcga_brca",
        )
        input_path = Path(os.path.realpath(__file__)).parent.resolve()
        self.dic = {
            "input_preprocessed": os.path.join(input_path, "brca.csv"),
            "train_test_split": os.path.join(
                input_path, "dataset_creation_scripts/train_test_split.csv"
            ),
        }
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.data = pd.read_csv(self.dic["input_preprocessed"])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data.iloc[idx, 1:40]
        y = self.data.iloc[idx, 40:42]
        return (torch.tensor(x, dtype=self.X_dtype), torch.tensor(y, dtype=self.y_dtype))


class FedTcgaBrca(TcgaBrcaRaw):
    """
    Pytorch dataset containing all the clinical features and (event, time)
    information for TCGA-BRCA survival analysis.
    One can instantiate this dataset with train or test data coming from either of
    the 6 regions or all regions pooled.
    The train/test split is static and given in the train_test_split file.

    Parameters
    ----------
    center : int, optional
        Between 0 and 5, designates the region in the case of pooled==False. Default to
        0
    train : bool, optional
        Characterizes if the dataset is used for training or for testing. Default to True
    pooled : bool, optional
        Characterizes if the dataset is pooled or not. Default to False
    X_dtype : torch.dtype, optional
        Default to torch.float32
    y_dtype : torch.dtype, optional
        Default to torch.float32
    """

    def __init__(
        self,
        center: int = 0,
        train: bool = True,
        pooled: bool = False,
        X_dtype: torch.dtype = torch.float32,
        y_dtype: torch.dtype = torch.float32,
    ):
        """
        cf class docstring
        """

        super().__init__(X_dtype=X_dtype, y_dtype=y_dtype)

        self.center = center
        self.train_test = "train" if train else "test"
        self.pooled = pooled
        self.key = self.train_test + "_" + str(self.center)
        df = pd.read_csv(self.dic["train_test_split"])

        if self.pooled:
            pids = df.query("fold == '" + self.train_test + "' ").reset_index(drop=True)

        if not self.pooled:
            assert center in range(6)
            pids = df.query("fold2 == '" + self.key + "' ").reset_index(drop=True)

        pid_list = list(pids["pid"])
        df2 = pd.read_csv(self.dic["input_preprocessed"])
        self.data = df2[df2["pid"].isin(pid_list)]


if __name__ == "__main__":

    mydataset = TcgaBrcaRaw()
    print(len(mydataset))
    print("Example of dataset record: ", mydataset[0])

    mydataset = FedTcgaBrca(train=True, pooled=True)
    print(len(mydataset))
    print("Example of dataset record: ", mydataset[0])
    mydataset = FedTcgaBrca(train=False, pooled=True)
    print(len(mydataset))
    print("Example of dataset record: ", mydataset[0])

    for i in range(6):
        mydataset = FedTcgaBrca(center=i, train=True, pooled=False)
        print(len(mydataset))
        print("Example of dataset record: ", mydataset[0])
        mydataset = FedTcgaBrca(center=i, train=False, pooled=False)
        print(len(mydataset))
        print("Example of dataset record: ", mydataset[0])

    mydataset = FedTcgaBrca(center=5, train=False, pooled=False)
    print(len(mydataset))
    for i in range(11):
        print("Example of dataset record: ", mydataset[i])
