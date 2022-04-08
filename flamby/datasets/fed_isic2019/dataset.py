import os
import random
from pathlib import Path

import albumentations
import numpy as np
import pandas as pd
import torch
from PIL import Image

from flamby.utils import check_dataset_from_config


class Isic2019Raw(torch.utils.data.Dataset):
    """Pytorch dataset containing all the features, labels and datacenter
    information for Isic2019.
    Attributes
    ----------
    image_paths: list[str], the list with the path towards all features
    targets: list[int], the list with all classification labels for all features
    centers: list[int], the list for all datacenters for all features
    X_dtype: torch.dtype, the dtype of the X features output
    y_dtype: torch.dtype, the dtype of the y label output
    train_test: str, characterizes if the dataset is used for training or for
    testing, equals "train" or "test"
    augmentations: image transform operations from the albumentations library,
    used for data augmentation
    dic: dictionary containing the paths to the input images and the
    train_test_split file
    """

    def __init__(
        self,
        train_test,
        X_dtype=torch.float32,
        y_dtype=torch.int64,
        augmentations=None,
    ):
        dict = check_dataset_from_config(dataset_name="fed_isic2019", debug=False)
        input_path = dict["dataset_path"]
        dir = str(Path(os.path.realpath(__file__)).parent.resolve())
        self.dic = {
            "input_preprocessed": os.path.join(
                input_path, "ISIC_2019_Training_Input_preprocessed"
            ),
            "train_test_split": os.path.join(
                dir, "dataset_creation_scripts/train_test_split"
            ),
        }
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.train_test = train_test
        df = pd.read_csv(self.dic["train_test_split"])
        df2 = df.query("fold == '" + self.train_test + "' ").reset_index(drop=True)
        images = df2.image.tolist()
        self.image_paths = [
            os.path.join(self.dic["input_preprocessed"], image_name + ".jpg")
            for image_name in images
        ]
        self.targets = df2.target
        self.augmentations = augmentations
        self.centers = df2.center

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path))
        target = self.targets[idx]

        # Image augmentations
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return (
            torch.tensor(image, dtype=self.X_dtype),
            torch.tensor(target, dtype=self.y_dtype),
        )


class FedIsic2019(Isic2019Raw):
    """
    Pytorch dataset containing for each center the features and associated labels
    for the Isic2019 federated classification.
    One can instantiate this dataset with train or test data coming from either of
    the 6 centers it was created from or all data pooled.
    The train/test split is fixed and given in the train_test_split file.
    Attributes
    ----------
    pooled: boolean, characterizes if the dataset is pooled or not
    center: int, between 0 and 5, designates the datacenter in the case of pooled==False
    """

    def __init__(
        self,
        center,
        pooled,
        train_test,
        X_dtype=torch.float32,
        y_dtype=torch.int64,
        augmentations=None,
    ):

        super().__init__(
            train_test,
            X_dtype=X_dtype,
            y_dtype=y_dtype,
            augmentations=augmentations,
        )

        self.center = center
        self.pooled = pooled
        key = self.train_test + "_" + str(self.center)
        if not self.pooled:
            assert center in range(6)
            df = pd.read_csv(self.dic["train_test_split"])
            df2 = df.query("fold2 == '" + key + "' ").reset_index(drop=True)
            images = df2.image.tolist()
            self.image_paths = [
                os.path.join(self.dic["input_preprocessed"], image_name + ".jpg")
                for image_name in images
            ]
            self.targets = df2.target
            self.centers = df2.center


if __name__ == "__main__":

    sz = 200
    train_aug = albumentations.Compose(
        [
            albumentations.RandomScale(0.07),
            albumentations.Rotate(50),
            albumentations.RandomBrightnessContrast(0.15, 0.1),
            albumentations.Flip(p=0.5),
            albumentations.Affine(shear=0.1),
            albumentations.RandomCrop(sz, sz) if sz else albumentations.NoOp(),
            albumentations.CoarseDropout(random.randint(1, 8), 16, 16),
            albumentations.Normalize(always_apply=True),
        ]
    )
    test_aug = albumentations.Compose(
        [
            albumentations.CenterCrop(sz, sz),
            albumentations.Normalize(always_apply=True),
        ]
    )

    mydataset = FedIsic2019(5, True, "test", augmentations=test_aug)

    print("Example of dataset record: ", mydataset[0])
    print(f"The dataset has {len(mydataset)} elements")
    for i in range(50):
        print(f"Size of image {i} ", mydataset[i][0].shape)
        print(f"Target {i} ", mydataset[i][1])
