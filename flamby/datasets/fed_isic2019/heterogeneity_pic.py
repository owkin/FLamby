import argparse
import copy
import os
import random
import time

import albumentations
import dataset
import torch
from sklearn import metrics

from flamby.datasets.fed_isic2019 import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    FedIsic2019,
    metric,
    NUM_CLIENTS
)
from flamby.utils import check_dataset_from_config, evaluate_model_on_tests

import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from flamby.datasets.fed_isic2019 import FedIsic2019

import umap.umap_ as umap
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


class model_eff_net_pretrained(nn.Module):
    
    def __init__(self, pretrained=True, arch_name="efficientnet-b0"):
        super(model_eff_net_pretrained, self).__init__()
        self.pretrained = pretrained
        self.base_model = (
            EfficientNet.from_pretrained(arch_name)
            if pretrained
            else EfficientNet.from_name(arch_name)
        )
        self.base_model._fc = nn.Sequential()

    def forward(self, image):
        out = self.base_model(image)
        return out


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--GPU",
        type=int,
        default=0,
        help="GPU to run the training on (if available)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Numbers of workers for the dataloader",
    )
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    torch.use_deterministic_algorithms(False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)

    model = model_eff_net_pretrained()
    model = model.to(device)
    model.eval()

    X = np.empty((0,1280))
    centers = np.empty((0,1))

    for i in range (NUM_CLIENTS):
        mydataset = FedIsic2019(center=i, train=True, pooled=False)
        dataloader = torch.utils.data.DataLoader(mydataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.workers)
        for sample in dataloader:
            inputs = sample[0].to(device)
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
            X_ = outputs.cpu().detach().clone().numpy()
            X = np.concatenate([X, X_])
            centers_ = i * np.ones((outputs.shape[0],1))
            centers = np.concatenate([centers, centers_])


    #pca = PCA(10)
    #X = pca.fit_transform(X)

    X_reduced = umap.UMAP().fit_transform(X)

    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=centers, s=0.5)
    plt.savefig('heterogeneity_pic.png')
