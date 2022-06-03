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
        plt.plot(bins[1:], bars, label=f"center {center}")
    plt.legend()
    plt.savefig("distribution_v2.png")
    plt.show()


def run_plot_sample_per_sample():
    list_x = np.linspace(0, 1, num=200)
    colors = ["red", "blue", "red", "red"]
    os.makedirs("per_sample", exist_ok=True)
    max_sample = 20
    # for center in [0, 1, 2, 3]:
    #     print(f"doing center {center}")
    #     ds = FedLidcIdri(
    #         train=True,
    #         pooled=False,
    #         center=center,
    #         debug=False,
    #     )
    #     for k in tqdm(range(min(len(ds), max_sample))):
    #         if not os.path.exists(f"per_sample/counts_center_{center}_sample_{k}.npy"):
    #             data = ds[k][0].detach().cpu().ravel()
    #             data = data[data > 0.0]
    #             data = data[data < 1.0]
    #             counts, bins, bars = plt.hist(
    #                 data, density=True, bins=list_x, alpha=0.5, label=f"center {k+1}"
    #             )
    #             np.save(f"per_sample/counts_center_{center}_sample_{k}.npy", counts)
    #     plt.clf()
    #     # np.save("per_sample/bins.npy", bins)

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
            # bars = np.minimum(bars, 15)
            if k == 0:
                plt.plot(
                    bins[1:][:80],
                    bars[:80],
                    color=colors[center],
                    linewidth=0.2,
                    alpha=0.5,
                    label=f"center {center}",
                )
            else:
                plt.plot(
                    bins[1:][:80],
                    bars[:80],
                    color=colors[center],
                    linewidth=0.5,
                    alpha=0.5,
                )
    plt.legend()
    # plt.savefig("distribution_per_sample.png")
    plt.show()


def run_plot_y():
    max_sample = 1000
    list_bins = np.linspace(0, 5000, num=50)
    list_counts = []
    for center in [0, 1, 2, 3]:
        # list_data = []
        # print(f"doing center {center}")
        # ds = FedLidcIdri(
        #     train=True,
        #     pooled=False,
        #     center=center,
        #     debug=False,
        # )
        # for k in tqdm(range(min(len(ds), max_sample))):
        #     data = np.sum(ds[k][1].detach().cpu().numpy().ravel())
        #     list_data.append(data)
        # np.save(f'ratio_y_center_{center}.npy', list_data)
        list_data = np.load(f"ratio_y_center_{center}.npy")
        counts, bins, bars = plt.hist(
            np.array(list_data),
            density=True,
            label=f"center {center+1}",
            alpha=0.5,
            bins=list_bins,
        )
        list_counts.append(counts)

    plt.legend()
    plt.title("Number of pixel with tumors")
    plt.savefig("distribution_y.png")
    plt.show()
    for count in list_counts:
        plt.plot(bins[1:], count)
    plt.title("Number of pixel with tumors")
    plt.show()


if __name__ == "__main__":
    # run_plot()
    # run_plot_sample_per_sample()
    run_plot_y()
