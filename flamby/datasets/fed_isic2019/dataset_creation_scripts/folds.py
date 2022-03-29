import os

import numpy as np
import pandas as pd
from sklearn import model_selection

dic = {
    "labels": "ISIC_2019_Training_GroundTruth.csv",
    "metadata_path": "ISIC_2019_Training_Metadata_FL.csv",
}

if __name__ == "__main__":

    input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    df = pd.read_csv(os.path.join(input_path, dic["labels"]))
    onehot = df[["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]].values
    df["target"] = [np.where(r == 1)[0][0] for r in onehot]
    df2 = pd.read_csv(os.path.join(input_path, dic["metadata_path"]))
    df["center"] = df2["dataset"]

    df = df.sample(frac=1).reset_index(drop=True)
    df = df.replace(
        [
            "BCN_nan",
            "HAM_vidir_molemax",
            "HAM_vidir_modern",
            "HAM_rosendahl",
            "MSK4nan",
            "HAM_vienna_dias",
        ],
        ["0", "1", "2", "3", "4", "5"],
    )

    X = df.image.values
    centers = df.center.values

    X_train_2, X_test_2 = model_selection.train_test_split(
        X, test_size=0.2, stratify=centers, random_state=13
    )
    for train_index in X_train_2:
        df.loc[df.image == train_index, "fold"] = "train"
        df.loc[df.image == train_index, "fold2"] = (
            "train" + "_" + df.loc[df.image == train_index, "center"]
        )
    for test_index in X_test_2:
        df.loc[df.image == test_index, "fold"] = "test"
        df.loc[df.image == test_index, "fold2"] = (
            "test" + "_" + df.loc[df.image == test_index, "center"]
        )

    df.to_csv(os.path.join(input_path, "train_test_folds.csv"), index=False)

    print("Number of images", df.shape[0])
    print("Class counts", df["target"].value_counts())
    print(
        "(Center, class) counts",
        df.groupby(["center", "target"]).size().unstack(fill_value=0),
    )
    print("Center counts", df["center"].value_counts())
    print("Pooled train/test split", df["fold"].value_counts())
    print("Stratified train/test split", df["fold2"].value_counts())
