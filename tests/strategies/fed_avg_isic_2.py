import random

import albumentations
import torch
from torch.utils.data import DataLoader as dl

from flamby.datasets.fed_isic2019 import (
    BATCH_SIZE,
    LR,
    Baseline,
    BaselineLoss,
    FedIsic2019,
    metric,
)
from flamby.strategies.fed_avg import FedAvg
from flamby.utils import evaluate_model_on_tests


def fedavg_Isic():

    random.seed(0)
    torch.manual_seed(0)

    sz = 200
    train_aug = albumentations.Compose(
        [
            albumentations.RandomScale(0.07),
            albumentations.Rotate(50),
            albumentations.RandomBrightnessContrast(0.15, 0.1),
            albumentations.Flip(p=0.5),
            albumentations.Affine(shear=0.1),
            albumentations.RandomCrop(sz, sz),
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

    training_dls = [
        dl(
            FedIsic2019(train=True, pooled=True, augmentations=train_aug),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
        )
        for i in range(1)
    ]
    test_dls = [
        dl(
            FedIsic2019(train=False, pooled=True, augmentations=test_aug),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            drop_last=True,
        )
        for i in range(1)
    ]

    loss = BaselineLoss(
        torch.FloatTensor(
            [5.5813, 2.0472, 7.0204, 26.1194, 9.5369, 101.0707, 92.5224, 38.3443]
        )
    )

    m = Baseline()
    print("Test metric before training", evaluate_model_on_tests(m, test_dls, metric))

    NUM_UPDATES = 1
    nrounds = (18597 // BATCH_SIZE) * 1 // NUM_UPDATES

    optimizer_class = torch.optim.Adam

    s = FedAvg(
        training_dls, m, loss, optimizer_class, LR, NUM_UPDATES, nrounds, log=False
    )
    m = s.run()

    print(evaluate_model_on_tests(m[0], test_dls, metric))


if __name__ == "__main__":

    fedavg_Isic()
