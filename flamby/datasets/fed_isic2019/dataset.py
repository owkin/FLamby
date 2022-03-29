import os
import random

import albumentations
import numpy as np
import pandas as pd
import PIL
import torch

dic = {
    "input_preprocessed": "./ISIC_2019_Training_Input_preprocessed",
    "train_test_folds": "./train_test_folds.csv",
}


class ISIC2019DatasetRaw(torch.utils.data.Dataset):
    def __init__(
        self,
        train_test_folds_csv_path,
        train_test,
        X_dtype=torch.float32,
        y_dtype=torch.int64,
        augmentations=None,
    ):
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.train_test = train_test
        df = pd.read_csv(train_test_folds_csv_path)
        df2 = df.query("fold == '" + self.train_test + "' ").reset_index(drop=True)
        images = df2.image.tolist()
        self.image_paths = [
            os.path.join(dic["input_preprocessed"], image_name + ".jpg")
            for image_name in images
        ]
        self.targets = df2.target
        self.augmentations = augmentations
        self.centers = df2.center

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.array(PIL.Image.open(image_path))
        target = self.targets[idx]

        # Image augmentations
        if (self.augmentations is not None) and (self.train_test == "train"):
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "image": torch.tensor(image, dtype=self.X_dtype),
            "target": torch.tensor(target, dtype=self.y_dtype),
        }


class FedISIC2019Dataset(ISIC2019DatasetRaw):
    def __init__(
        self,
        center,
        pooled,
        train_test_folds_csv_path,
        train_test,
        X_dtype=torch.float32,
        y_dtype=torch.int64,
        augmentations=None,
    ):

        super().__init__(
            train_test_folds_csv_path,
            train_test,
            X_dtype=torch.float32,
            y_dtype=torch.int64,
            augmentations=None,
        )

        self.center = center
        self.pooled = pooled
        key = self.train_test + "_" + str(self.center)
        if not self.pooled:
            assert center in range(6)
            df = pd.read_csv(train_test_folds_csv_path)
            df2 = df.query("fold2 == '" + key + "' ").reset_index(drop=True)
            images = df2.image.tolist()
            self.image_paths = [
                os.path.join(dic["input_preprocessed"], image_name + ".jpg")
                for image_name in images
            ]
            self.targets = df2.target
            self.centers = df2.center


if __name__ == "__main__":

    sz = 384
    train_aug = albumentations.Compose(
        [
            albumentations.RandomScale(0.07),
            albumentations.Rotate(50),
            albumentations.RandomBrightnessContrast(0.15, 0.1),
            albumentations.Flip(p=0.5),
            albumentations.IAAAffine(shear=0.1),
            albumentations.RandomCrop(sz, sz) if sz else albumentations.NoOp(),
            albumentations.OneOf(
                [
                    albumentations.Cutout(random.randint(1, 8), 16, 16),
                    albumentations.CoarseDropout(random.randint(1, 8), 16, 16),
                ]
            ),
            albumentations.Normalize(always_apply=True),
        ]
    )

    mydataset = FedISIC2019Dataset(
        0, True, dic["train_test_folds"], "train", augmentations=train_aug
    )
    print(mydataset[0])
    print(len(mydataset))
    for i in range(50):
        print(mydataset[i]["image"].shape)
        print(mydataset[i]["target"])
