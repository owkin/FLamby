import argparse
import random

import albumentations
import dataset
import torch
import torch.nn as nn
import torchvision
from efficientnet_pytorch import EfficientNet
from loss import WeightedFocalLoss


class EfficientNetBx(nn.Module):
    def __init__(self, pretrained=True, arch_name="efficientnet-b0"):
        super(EfficientNetBx, self).__init__()
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

    def forward(self, image, target, weights=None, args=None):
        out = self.base_model(image)
        if args.loss == "weighted_focal_loss":
            loss = WeightedFocalLoss(alpha=weights)(out, target.view(-1, 1).type_as(out))
        elif args.loss == "crossentropy":
            loss = nn.CrossEntropyLoss(weight=weights)(out, target)
        else:
            raise ValueError("loss function not found.")
        return out, loss


if __name__ == "__main__":

    print(torch.__version__)
    print(torchvision.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", default="crossentropy")
    args = parser.parse_args()

    sz = 100
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

    dic = {
        "input_preprocessed": "./ISIC_2019_Training_Input_preprocessed",
        "train_test_folds": "./train_test_folds.csv",
    }

    mydataset = dataset.FedISIC2019(
        0, True, dic["train_test_folds"], "train", augmentations=train_aug
    )

    model = EfficientNetBx()

    for i in range(50):
        X = torch.unsqueeze(mydataset[i]["image"], 0)
        y = torch.unsqueeze(mydataset[i]["target"], 0)
        print(X.shape)
        print(y.shape)
        print(model(X, y, args=args))
