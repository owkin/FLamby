import inspect
import json
import os
from glob import glob

import numpy as np
import pandas as pd
import torch

dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results")
csv_files = glob(os.path.join(dir_path, "results_*.csv"))

dataset_names = [
    "_".join(csvf.split("/")[-1].split(".")[0].split("_")[2:]) for csvf in csv_files
]
optimizers_classes = [e[1] for e in inspect.getmembers(torch.optim, inspect.isclass)]
csvs = [pd.read_csv(e) for e in csv_files]
configs = []
for dname, csv, csvf in zip(dataset_names, csvs, csv_files):
    config = {}
    config["dataset"] = dname
    config["results_file"] = csvf.split("/")[-1]
    config["strategies"] = {}
    for stratname in [
        "Scaffold",
        "Cyclic",
        "FedAdam",
        "FedYogi",
        "FedAvg",
        "FedProx",
        "FedAdagrad",
    ]:
        config["strategies"][stratname] = {}
        current = csv.loc[
            (csv["Method"] == stratname + "100") & (csv["Test"] == "Pooled Test")
        ]
        current = current.reset_index()
        try:
            idx = current["Metric"].idxmax()
        except ValueError:
            print(f"For dataset {dname} missing {stratname} !!!")
            continue
        best_hyperparams = current.iloc[idx][
            [col for col in current.columns if col not in ["Test", "Method", "Metric"]]
        ].to_dict()
        best_hyperparams.pop("index")
        for k, v in best_hyperparams.items():
            try:
                isnan = np.isnan(v)
            except TypeError:
                isnan = False
            if not (isnan):
                has_corresp_opt = [
                    str(v) == str(opt_class) for opt_class in optimizers_classes
                ]

                if any(has_corresp_opt):
                    v = (
                        "torch.optim."
                        + optimizers_classes[has_corresp_opt.index(True)].__name__
                    )
                config["strategies"][stratname][k] = v

    with open(f"config_{dname}.json", "w") as outfile:
        json.dump(config, outfile, indent=4, sort_keys=True)
