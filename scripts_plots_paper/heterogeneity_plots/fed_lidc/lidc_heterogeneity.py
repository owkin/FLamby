import numpy as np
import pandas as pd
import seaborn as sns
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
    plt.clf()

    sns.set_theme(style="darkgrid")

    dict_center = {0: "GE MEDICAL SYSTEMS", 1: "PHILIPS", 2: "SIEMENS", 3: "TOSHIBA"}

    df = pd.DataFrame()
    for center in [0, 1, 2, 3]:
        bins = np.load("bins.npy")
        bars = np.load(f"counts_center_{center}.npy")
        df[dict_center[center]] = bars
    df["Intensity"] = bins[1:]
    df = df.set_index("Intensity")

    sns.lineplot(data=df)
    plt.xlabel("Voxels of intensity...")
    plt.ylabel("Density of voxels")
    plt.legend(prop={"size": 8})
    plt.savefig("fed_lidc_intensity.pdf")
    plt.show()


if __name__ == "__main__":
    run_plot()
