import json
import os
from glob import glob

import numpy as np
import pandas as pd

dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "flamby", "results")
breakpoint()
csv_files = glob(os.path.join(dir_path, "results_*.csv"))

dataset_names = [
    "_".join(csvf.split("/")[-1].split(".")[0].split("_")[2:]) for csvf in csv_files
]
csvs = [pd.read_csv(e) for e in csv_files]
configs = []
for dname, csv, csvf in zip(dataset_names, csvs, csv_files):
    config = {}
    config["dataset"] = dname
    config["results_file"] = csvf.split("/")[-1]
    config["strategies"] = {}
    current_xps_dicts = []
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
                config["strategies"][stratname][k] = v

        # Match best hyperparameters in original df and keep only those ones
        found_xps = csv[list(best_hyperparams)]
        found_xps_numerical = found_xps.select_dtypes(exclude=[object])

        col_numericals = found_xps_numerical.columns
        col_objects = [c for c in found_xps.columns if not (c in col_numericals)]

        if len(col_numericals) > 0:
            bool_numerical = np.all(
                np.isclose(
                    found_xps_numerical,
                    pd.Series(
                        {
                            k: float(best_hyperparams[k])
                            for k in list(best_hyperparams.keys())
                            if k in col_numericals
                        }
                    ),
                    equal_nan=True,
                ),
                axis=1,
            )
        else:
            bool_numerical = np.ones((len(csv.index), 1)).astype("bool")

        if len(col_objects):
            bool_objects = found_xps[col_objects].astype(str) == pd.Series(
                {
                    k: str(best_hyperparams[k])
                    for k in list(best_hyperparams.keys())
                    if k in col_objects
                }
            )
        else:
            bool_objects = np.ones((len(csv.index), 1)).astype("bool")

        bool_method = csv["Method"] == (stratname + str(100))
        index_of_interest_1 = csv.loc[pd.DataFrame(bool_numerical).all(axis=1)].index
        index_of_interest_2 = csv.loc[pd.DataFrame(bool_objects).all(axis=1)].index
        index_of_interest_3 = csv.loc[pd.DataFrame(bool_method).all(axis=1)].index
        index_of_interest = index_of_interest_1.intersection(
            index_of_interest_2
        ).intersection(index_of_interest_3)
        current_xps_dicts += csv.iloc[index_of_interest].to_dict("records")
    # Adding single-centric baselines
    current_xps_dicts += csv.loc[
        (csv["Method"] == "Pooled Training")
        | (["Local" in m for m in csv["Method"].tolist()])
    ].to_dict("records")

    with open(f"config_{dname}.json", "w") as outfile:
        json.dump(config, outfile)

    final_df = pd.DataFrame.from_dict(current_xps_dicts)
    final_df.to_csv(f"results_{dname}_v2.csv", index=False)
