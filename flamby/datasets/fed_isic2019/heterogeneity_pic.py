import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import umap.umap_ as umap
from efficientnet_pytorch import EfficientNet

from flamby.datasets.fed_isic2019 import BATCH_SIZE, NUM_CLIENTS, FedIsic2019


sns.set()


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

    X = np.empty((0, 1280))
    centers = np.empty((0, 1))
    colors = sns.color_palette("hls", 6)
    for i in range(NUM_CLIENTS):
        mydataset = FedIsic2019(center=i, train=True, pooled=False)
        dataloader = torch.utils.data.DataLoader(
            mydataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.workers
        )
        for sample in dataloader:
            inputs = sample[0].to(device)
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
            X_ = outputs.cpu().detach().clone().numpy()
            X = np.concatenate([X, X_])
            centers_ = i * np.ones((outputs.shape[0], 1))
            centers = np.concatenate([centers, centers_])

    def draw_umap(n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean"):
        print(
            f"Computing UMAP, NN {n_neighbors}, min d {min_dist}, ncomp {n_components}"
        )
        u = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
        ).fit_transform(X)
        if n_components == 1:
            fig = plt.figure()
            ax = fig.add_subplot()
            for i in range(NUM_CLIENTS):
                from_current_center = (centers == i)[:, 0]
                ax.scatter(
                    u[from_current_center, 0],
                    range(from_current_center.sum()),
                    color=colors[i],
                    s=0.5,
                    label=f"Local {i}",
                )
        if n_components == 2:
            fig = plt.figure()
            ax = fig.add_subplot()
            for i in range(NUM_CLIENTS):
                from_current_center = (centers == i)[:, 0]
                ax.scatter(
                    u[from_current_center, 0],
                    u[from_current_center, 1],
                    color=colors[i],
                    s=0.5,
                    label=f"Local {i}",
                )
            plt.xlabel("Umap dimension 1")
            plt.ylabel("Umap dimension 2")

        if n_components == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            for i in range(NUM_CLIENTS):
                from_current_center = (centers == i)[:, 0]
                ax.scatter(
                    u[from_current_center, 0],
                    u[from_current_center, 1],
                    u[from_current_center, 2],
                    color=colors[i],
                    s=0.5,
                    label=f"Local {i}",
                )
        lgnd = plt.legend()
        for handle in lgnd.legendHandles:
            handle.set_sizes([10.0])
        name_fig = (
            "heterogeneity_pic_"
            + str(n_neighbors)
            + "_"
            + str(min_dist)
            + "_"
            + str(n_components)
            + ".png"
        )
        name_fig_2 = (
            "heterogeneity_pic_"
            + str(n_neighbors)
            + "_"
            + str(min_dist)
            + "_"
            + str(n_components)
            + ".eps"
        )
        print(f"Saving {name_fig}")
        plt.savefig(name_fig)
        plt.savefig(name_fig_2)

    draw_umap(n_neighbors=250, min_dist=0.5, n_components=2)
