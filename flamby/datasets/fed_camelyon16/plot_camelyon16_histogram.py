import matplotlib.pyplot as plt
import numpy as np

from flamby.utils import seaborn_styling

seaborn_styling(figsize=(15, 10))

if __name__ == "__main__":
    # Load the histograms
    histogram0 = np.load("histogram_0.npy")
    bins = np.load("bins_0.npy")
    histogram1 = np.load("histogram_1.npy")

    colors = ["red", "green", "blue"]
    hatches = ["-", "/", "|||"]

    fig, axarr = plt.subplots(2, figsize=(12, 8), sharex=True)

    for id_center, histogram in enumerate([histogram0, histogram1]):

        for i, (c, h) in enumerate(zip(colors, hatches)):
            axarr[id_center].hist(
                bins[:-1],
                bins,
                weights=histogram[i],
                color=c,
                hatch=h,
                alpha=0.5,
                histtype="stepfilled",
                label=c,
            )
        axarr[id_center].set_ylim(0, 0.013)
        axarr[id_center].legend(loc="upper left", frameon=False)
        axarr[id_center].set_title(f"Client {id_center}")
        axarr[id_center].title.set_size(24)
    axarr[1].set_xlabel("Intensity")
    plt.savefig("heterogeneity_fed_camelyon16.png")
