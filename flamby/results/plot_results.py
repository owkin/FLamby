import os
import re
from glob import glob

import matplotlib.pyplot as plt
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

csv_files = glob(os.path.join(dir_path, "results_*.csv"))

# Assumes the csv file name follows results_benchmark_fed_my_dataset convention will probably break for IXITiny make sure filename has fed_ixi
dataset_names = [
    "_".join(csvf.split("/")[-1].split(".")[0].split("_")[2:]) for csvf in csv_files
]
results = [pd.read_csv(csvf) for csvf in csv_files]


fig, axs = plt.subplots(2, 4, sharey=True, figsize=(40, 13), dpi=80)
# Keep Room for Strategy labels
fig.subplots_adjust(hspace=0.5)
flattened_axs = axs.flatten()


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

    for m in current_methods:
        assert len(res.loc[res["Method"] == m].index) == current_num_clients

    # Pretiifying render
    res = res.rename(columns={"Method": "Training Method"})
    res.loc[res["Training Method"] == "Pooled Training", "Training Method"] = "Pooled"
    current_methods_display = [re.sub(" Training", "", c) for c in current_methods]
    sns.barplot(
        ax=ax,
        x="Training Method",
        y="Metric",
        data=res,
        order=current_methods_display,
        capsize=0.05,
        saturation=8,
        errcolor="gray",
        errwidth=2,
    )
    # We remove the number of updates which will go into the legend
    labels = [re.sub("100", "", item.get_text()) for item in ax.get_xticklabels()]

    # TODO cannot make fontsize change effective on those labels for some reasons
    ax.set_xticklabels(labels, rotation=90, fontsize=20)
    ax.set_xlabel(ax.get_xlabel(), fontsize=20)
    ax.set_ylabel(ax.get_ylabel(), fontsize=20)

    # We only display the xlabel on figures from the second row except the 4th one because it has no counterpart(label can be removed entirely)
    # and y label on the first figure of each row
    if idx < 3:
        ax.set(xlabel=None)
    if idx not in [0, 4]:
        ax.set(ylabel=None)
    current_title = " ".join([word.capitalize() for word in name.split("_")])

    ax.set_title(current_title, fontsize=20, fontweight="bold")

# Hide the last plot
flattened_axs[-1].set_visible(False)
plt.tight_layout()
plt.savefig("plot_results_benchmarks.png")
