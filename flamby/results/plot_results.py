import os
import re
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()


# utils to fetch number of clients from each dataset names
def get_nb_clients_from_dataset(name_arg):
    print(name_arg)
    return getattr(
        __import__(f"flamby.datasets.{name_arg}", fromlist=["NUM_CLIENTS"]),
        "NUM_CLIENTS",
    )


strategies_names = [
    "FedAvg",
    "Scaffold",
    "FedProx",
    "Cyclic",
    "FedAdagrad",
    "FedYogi",
    "FedAdam",
]
# Filtering only 100 updates strategies
strategies = [strat + str(100) for strat in strategies_names]

dir_path = os.path.dirname(os.path.realpath(__file__))

# TODO remove only there for debugging purposes
# for name in ["fed_lidc_idri", "fed_ixi", "fed_camelyon16"]:
#     current_num_clients = get_nb_clients_from_dataset(name)
#     methods = strategies + ['Pooled Training'] +  ["Local " + str(i) for i in range(current_num_clients)]
#     # duplicate methods
#     methods = [m for m in methods for _ in range(current_num_clients)]
#     nrows = len(methods)
#     metrics = np.random.uniform(0, 1, nrows)
#     tests = [""] * nrows
#     df = pd.DataFrame({"Method": methods, "Metric": metrics, "Test": tests})
#     df.to_csv(os.path.join(dir_path, f"results_benchmark_{name}.csv"), index=False)

# ugly but needed to fix the order of the plots quickly
# csv_files = glob(os.path.join(dir_path, "results_*.csv"))
csv_files = [
            os.path.join(dir_path, "results_benchmark_fed_camelyon16.csv"),
            os.path.join(dir_path, "results_benchmark_fed_lidc_idri.csv"),
            os.path.join(dir_path, "results_benchmark_fed_ixi.csv"),
            os.path.join(dir_path, "results_benchmark_fed_tcga_brca.csv"),
            os.path.join(dir_path, "results_benchmark_fed_kits19.csv"),
            os.path.join(dir_path, "results_benchmark_fed_isic2019.csv"),
            os.path.join(dir_path, "results_benchmark_fed_heart_disease.csv"),
            ]

# Assumes the csv file name follows results_benchmark_fed_my_dataset convention will probably break for IXITiny make sure filename has fed_ixi
# For lidc, filename should be fed_lidc_idri
dataset_names = [
    "_".join(csvf.split("/")[-1].split(".")[0].split("_")[2:]) for csvf in csv_files
]
results = [pd.read_csv(csvf) for csvf in csv_files]


fig, axs = plt.subplots(2, 4, sharey=True, figsize=(40, 13), dpi=80)
# Keep Room for Strategy labels
#fig.subplots_adjust(hspace=0.5)
fig.subplots_adjust(hspace=5.0)
flattened_axs = axs.flatten()

#palette = sns.color_palette("hls", 14)
palette = sns.color_palette("mako")
for idx, (ax, res, name) in enumerate(zip(flattened_axs, results, dataset_names)):
    # Remove pooled results
    res = res.loc[res["Test"] != "Pooled Test"]
    res = res[["Method", "Metric"]]

    current_num_clients = get_nb_clients_from_dataset(name)
    current_methods = (
        ["Pooled Training"]
        + ["Local " + str(i) for i in range(current_num_clients)]
        + strategies
    )

    res = res.loc[res["Method"].isin(current_methods)]
    # TODO remove try except when we have full results
    try:
        assert len(res["Method"].unique()) == len(current_methods)
    except AssertionError:
        for c in current_methods:
            if c not in res["Method"].unique().tolist():
                print(c)
                dictdf = res.to_dict("records")
                for i in range(current_num_clients):
                    dictdf.append({"Method": c, "Metric": np.nan})
                res = pd.DataFrame.from_dict(dictdf)

    #for m in current_methods:
    #    assert len(res.loc[res["Method"] == m].index) == current_num_clients

    # Pretiifying render
    res = res.rename(columns={"Method": "Training Method"})
    res.loc[res["Training Method"] == "Pooled Training", "Training Method"] = "Pooled"
    current_methods_display = [re.sub(" Training", "", c) for c in current_methods]

    # Messing with palettes to keep the same color for pooled and strategies
    current_palette = [palette[0]] + palette[1 : (current_num_clients + 1)] + palette[7:]
    sns.barplot(
        ax=ax,
        x="Training Method",
        y="Metric",
        data=res,
        order=current_methods_display,
        capsize=0.05,
        #saturation=8,
        saturation=2,
        errcolor="gray",
        errwidth=2,
        palette=current_palette,
    )
    # We remove the number of updates which will go into the legend
    labels = [re.sub("100", "", item.get_text()) for item in ax.get_xticklabels()]

    # TODO cannot make fontsize change effective on those labels for some reasons

    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
    ticks_loc = ax.get_yticks().tolist()

    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    label_format = "{:.1f}"
    ax.set_yticklabels([label_format.format(y) for y in ticks_loc], fontsize=25)

    ax.set_xticklabels(labels, rotation=90, fontsize=25, fontweight="heavy")
    ax.set_xlabel(ax.get_xlabel(), fontsize=25, fontweight="heavy")
    ax.set_ylabel(ax.get_ylabel(), fontsize=25, fontweight="heavy")

    # We only display the xlabel on figures from the second row except the 4th one because it has no counterpart(label can be removed entirely)
    # and y label on the first figure of each row
    if idx < 3:
        ax.set(xlabel=None)
    if idx not in [0, 4]:
        ax.set(ylabel=None)

    # ugly but no time
    #current_title = " ".join([word.capitalize() for word in name.split("_")])
    title_dicts = {
                'fed_camelyon16': 'Fed-Camelyon16',
                'fed_lidc_idri': 'Fed-LIDC-IDRI',
                'fed_ixi': 'Fed-IXI',
                'fed_tcga_brca': 'Fed-TCGA-BRCA',
                'fed_kits19': 'Fed-KITS2019',
                'fed_isic2019': 'Fed-ISIC2019',
                'fed_heart_disease': 'Fed-Heart-Disease',
                }
    current_title = title_dicts[name]
    ax.set_title(current_title, fontsize=35, fontweight="heavy")

# Hide the last plot
flattened_axs[-1].set_visible(False)
plt.tight_layout()
plt.savefig("plot_results_benchmarks.png")
plt.savefig("plot_results_benchmarks.eps")
