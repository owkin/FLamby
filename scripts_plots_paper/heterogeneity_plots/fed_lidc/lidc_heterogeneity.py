import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from flamby.datasets.fed_lidc_idri import FedLidcIdri


def run_plot():
    list_x = np.linspace(0, 1, num=200)
    for center in [0, 1, 2, 3]:
        print(f"doing center {center}")
        ds = FedLidcIdri(
            train=True,
            pooled=False,
            center=center,
            debug=False,
        )
        list_data = []
        for k in tqdm(range(len(ds))):
            data = ds[k][0].detach().cpu().ravel()
            data = data[data > 0.0]
            data = data[data < 1.0]
            list_data.append(data)
        list_data = np.concatenate(list_data)
        counts, bins, bars = plt.hist(
            list_data, density=True, bins=list_x, alpha=0.5, label=f"center {k+1}"
        )
        np.save("bins.npy", bins)
        np.save(f"counts_center_{center}.npy", counts)
    plt.savefig("distribution.png")
    plt.show()
    plt.clf()

    for center in [0, 1, 2, 3]:
        bins = np.load("bins.npy")
        bars = np.load(f"counts_center_{center}.npy")
        plt.plot(bins[1:], bars)
    plt.savefig("distribution_v2.png")
    plt.show()

def run_plot_sample_per_sample():
    list_x = np.linspace(0, 1, num=200)
    colors = ['red', 'blue', 'green', 'black']
    os.makedirs("per_sample", exist_ok=True)
    max_sample = 20
    for center in [0, 1, 2, 3]:
        print(f"doing center {center}")
        ds = FedLidcIdri(
            train=True,
            pooled=False,
            center=center,
            debug=False,
        )
        for k in tqdm(range(min(len(ds), max_sample))):
            data = ds[k][0].detach().cpu().ravel()
            data = data[data > 0.0]
            data = data[data < 1.0]
            counts, bins, bars = plt.hist(
                data, density=True, bins=list_x, alpha=0.5, label=f"center {k+1}"
            )
            np.save(f"per_sample/counts_center_{center}_sample_{k}.npy", counts)
        plt.clf()
        np.save("per_sample/bins.npy", bins)

    plt.clf()
    for center in [0, 1, 2, 3]:
        ds = FedLidcIdri(
            train=True,
            pooled=False,
            center=center,
            debug=False,
        )
        for k in range(min(len(ds), max_sample)):
            bins = np.load("per_sample/bins.npy")
            bars = np.load(f"per_sample/counts_center_{center}_sample_{k}.npy")
            bars = np.minimum(bars, 15)
            plt.plot(bins[1:], bars, color=colors[center], linewidth=0.2, alpha=0.5)
    plt.savefig("distribution_per_sample.png")
    plt.show()


if __name__ == "__main__":
    # run_plot()
    run_plot_sample_per_sample()
