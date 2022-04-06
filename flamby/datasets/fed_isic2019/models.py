# We use EfficientNet + a linear layer as a baseline model
# Thank you to [Luke Melas-Kyriazi](https://github.com/lukemelas) for his
# [pytorch reimplementation of EfficientNets]
# (https://github.com/lukemelas/EfficientNet-PyTorch)

import random

import albumentations
import dataset
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class Baseline(nn.Module):
    def __init__(self, pretrained=True, arch_name="efficientnet-b0"):
        super(Baseline, self).__init__()
        self.pretrained = pretrained
        self.base_model = (
            EfficientNet.from_pretrained(arch_name)
            if pretrained
            else EfficientNet.from_name(arch_name)
        )
        # self.base_model=torchvision.models.efficientnet_v2_s(pretrained=pretrained)
        nftrs = self.base_model._fc.in_features
        print("Number of features output by EfficientNet", nftrs)
        self.base_model._fc = nn.Linear(nftrs, 8)

    def forward(self, image):
        out = self.base_model(image)
        return out


if __name__ == "__main__":

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

    mydataset = dataset.FedIsic2019(0, True, "train", augmentations=train_aug)

    model = Baseline()

    for i in range(50):
        X = torch.unsqueeze(mydataset[i]["image"], 0)
        y = torch.unsqueeze(mydataset[i]["target"], 0)
        print(X.shape)
        print(y.shape)
        print(model(X))
