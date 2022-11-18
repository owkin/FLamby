import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns
from batchgenerators.utilities.file_and_folder_operations import join
from matplotlib.lines import Line2D
from numpy import random

from flamby.utils import get_config_file_path, read_config, seaborn_styling

seaborn_styling(figsize=(15, 10), legend_fontsize=20)


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Specify if debug mode (True) or not (False)",
    )
    args = parser.parse_args()
    return args


def plot_histogram(axis, array, num_positions=100, label=None, alpha=0.5, color=None):
    kwargs = dict(linewidth=1, color="black" if color is None else color, alpha=alpha)
    # RAM and computation issues force us to subsample
    values = random.choice(array, size=min(500000, array.shape[0]), replace=False)
    del array
    sns.kdeplot(
        values,
        color=kwargs["color"],
        fill=kwargs["color"],
        alpha=0.17,
        label=label,
        ax=axis,
    )


def read_csv_file_for_plotting(csv_file="../metadata/anony_sites.csv", base=None):
    print(" Reading kits19 Meta Data ...")
    columns = defaultdict(list)  # each value in each column is appended to a list

    with open(csv_file) as f:
        reader = csv.DictReader(f)  # read rows into a dictionary format
        for row in reader:  # read a row as {column1: value1, column2: value2,...}
            for (k, v) in row.items():  # go over each column name and value
                if k == "case":
                    columns[k].append(v)  # append the value into the appropriate list
                    # based on column name k
                else:
                    columns[k].append(int(v))

    case_ids = columns["case"]
    site_ids = columns["site_id"]

    train_case_ids = case_ids[0:210]
    train_site_ids = site_ids[0:210]

    print(" Creating Heterogeneity Plot ")
    silo_count = 0
    colours = sns.color_palette("colorblind", 6)

    fig, ax = plt.subplots()
    for ID in range(0, 89):
        client_ids = np.where(np.array(train_site_ids) == ID)[0]
        if len(client_ids) >= 10:
            client_data_idxx = np.array([train_case_ids[i] for i in client_ids])
            print("Silo ID " + str(silo_count))
            print(client_data_idxx)
            silo_array = np.zeros((0,), dtype="float32")
            for CI in client_data_idxx:
                print("Plotting ID " + str(CI))
                curr = join(base, CI)
                image_file = join(curr, "imaging.nii.gz")
                array = np.array(nib.load(image_file).dataobj).astype("float32")
                # We have finite RAM unfortunately
                if silo_array.shape[0] > 1e9:
                    break
                silo_array = np.concatenate((silo_array, array.ravel()))
            plot_histogram(ax, silo_array, color=colours[silo_count])
            silo_count += 1

    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    ax.tick_params(axis="x", labelsize=19)
    ax.tick_params(axis="y", labelsize=19)
    legend_elements = [
        Line2D([0], [0], color=colours[0], lw=4, label="Client 0"),
        Line2D([0], [0], color=colours[1], lw=4, label="Client 1"),
        Line2D([0], [0], color=colours[2], lw=4, label="Client 2"),
        Line2D([0], [0], color=colours[3], lw=4, label="Client 3"),
        Line2D([0], [0], color=colours[4], lw=4, label="Client 4"),
        Line2D([0], [0], color=colours[5], lw=4, label="Client 5"),
    ]

    ax.legend(handles=legend_elements, loc="upper right")
    plt.savefig("histograms_kits19.pdf", bbox_inches="tight")


if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    path_to_config_file = get_config_file_path("fed_kits19", debug=args.debug)
    dictt = read_config(path_to_config_file)
    base = dictt["dataset_path"] + "/data"
    print(base)
    read_csv_file_for_plotting(base=base)
