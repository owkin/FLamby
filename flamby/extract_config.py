import argparse
import inspect
import json
import os

import numpy as np
import pandas as pd
import torch


def main(args_cli):
    datasets = [
        "fed_kits19",
        "fed_ixi",
        "fed_camelyon16",
        "fed_isic2019",
        "fed_lidc_idri",
        "fed_heart_disease",
        "fed_tcga_brca",
    ]

    csv_files = args_cli.path_to_results
    if args_cli.dataset_name is None:
        dataset_names = [
            "_".join(csvf.split("/")[-1].split(".")[0].split("_")[2:])
            for csvf in csv_files
        ]
        assert all([d in datasets for d in dataset_names])
    else:
        if len(args_cli.dataset_name) == len(csv_files):
            dataset_names = args_cli.dataset_name
        elif len(args_cli.dataset_name) == 1:
            dataset_names = [args_cli.dataset_name[0] for _ in range(len(csv_files))]
        else:
            raise ValueError(
                "You should provide as many dataset names as you gave results"
                " files or 1 if they all come from the same dataset."
            )
    optimizers_classes = [e[1] for e in inspect.getmembers(torch.optim, inspect.isclass)]
    csvs = [pd.read_csv(e) for e in csv_files]
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
                [
                    col
                    for col in current.columns
                    if col not in ["Test", "Method", "Metric"]
                ]
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
        results_file_basename = csvf.split("/")[-1].split(".")[0]
        root = f"config_{results_file_basename}"
        basename = root + ".json"
        c = 0
        while os.path.exists(os.path.join(args_cli.extract_to_path, basename)):
            basename = root + f"_{c}.json"
            c += 1
        with open(os.path.join(args_cli.extract_to_path, basename), "w") as outfile:
            json.dump(config, outfile, indent=4, sort_keys=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-to-results",
        type=str,
        default="./results/results.csv",
        nargs="+",
        help="The path of the file to extract config from.",
    )
    parser.add_argument(
        "--extract-to-path",
        type=str,
        default=".",
        help="The path where the config will be extracted",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="The dataset name of the associated results file."
        "If not provided tries to extract it from the results file name.",
        nargs="+",
        choices=[
            None,
            "fed_kits19",
            "fed_ixi",
            "fed_camelyon16",
            "fed_isic2019",
            "fed_lidc_idri",
            "fed_heart_disease",
            "fed_tcga_brca",
        ],
    )
    args = parser.parse_args()
    assert os.path.isdir(
        args.extract_to_path
    ), "You should provide a path towards a directory"
    main(args)
