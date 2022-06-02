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
            list_data.append(data)
        list_data = np.concatenate(list_data)
        plt.hist(list_data, density=True, bins=list_x, alpha=0.5, label=f"center {k+1}")
    plt.savefig("distribution.png")
    plt.show()


if __name__ == "__main__":
    run_plot()
