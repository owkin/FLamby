import inspect
import itertools
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import flamby

STRATS_NAMES = ["FedAvg", "FedProx", "FedAdam", "FedYogi", "FedAdagrad", "Cyclic"]
optimizers_classes = [e[1] for e in inspect.getmembers(torch.optim, inspect.isclass)]


def cleanup(files_list):
    [os.remove(file) for file in files_list]


def assert_dfs_equal(pair0, pair1, ignore_columns=[]):
    """This checks that two dataframes are equal

    Parameters
    ----------
    pair0 : pandas.DataFrame
        The first dataframe of the equality test
    pair1 : pandas.DataFrame
        The second dataframe of the equality test
    ignore_columns : list, optional
        The columns that comparison should exclude, by default []
    """
    ignore_columns = [col for col in ignore_columns if (col in pair0) and (col in pair1)]
    df1 = pair0.drop(columns=ignore_columns).fillna("-9")
    df2 = pair1.drop(columns=ignore_columns).fillna("-9")[df1.columns]
    assert (
        df1.drop(columns=["Metric"])
        .reset_index(drop=True)
        .equals(df2.drop(columns=["Metric"]).reset_index(drop=True))
    )
    assert np.allclose(df1["Metric"], df2["Metric"])


def seeding_performance_assert(dataset_name, nrep=2):
    """Test if repeating the same xp multiple times gives the exact same results.

    Parameters
    ----------
    dataset_name : str
        the name of the dataset to test.
    nrep : int, optional
        the number of times we repeat, by default 5
    """
    repetitions, filenames, _ = launch_all_xps_from_config(
        dataset_name=dataset_name, nrep=nrep
    )
    for sidx in range(len(repetitions[0])):
        all_dfs_for_seed = [rep[sidx] for rep in repetitions]
        paired_df = itertools.combinations(all_dfs_for_seed, 2)
        for pair in paired_df:
            assert_dfs_equal(pair[0], pair[1])

    cleanup(filenames)


def compare_single_centric_and_strategy_vs_all(dataset_name):
    """Test if using --strategy or --singe-centric-baseline with fed_benchmark
    gives the same results as when executing from a config file.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to test.

    """
    results, rfile_paths, cfp = launch_all_xps_from_config(
        dataset_name=dataset_name, nrep=1
    )
    results = results[0]

    def robust_isnan(arg):
        try:
            return np.isnan(arg)
        except TypeError:
            return False

    for r in results:
        for strat in STRATS_NAMES:
            strat_res = r.loc[r["Method"] == strat + "100"]
            param_row = strat_res.iloc[0].to_dict()
            param_row = {k: v for k, v in param_row.items() if not (robust_isnan(v))}
            s = param_row.pop("seed")
            param_row.pop("Metric")
            param_row.pop("Test")
            param_row.pop("Method")
            tmp_results_file = f"tmp_strategy{strat}_seed{s}.csv"
            cmd = Path(flamby.__file__).parent / "benchmarks/fed_benchmark.py"
            cmd = "yes | python " + str(cmd)
            cmd += f" --seed {s} -cfp {cfp} -rfp {tmp_results_file}"
            cmd += f" --strategy {strat}"
            for k, v in param_row.items():
                if k in [
                    "learning_rate",
                    "server_learning_rate",
                    "mu",
                    "optimizer_class",
                    "beta1",
                    "beta2",
                    "tau",
                ]:
                    has_corresp_opt = [
                        str(v) == str(opt_class) for opt_class in optimizers_classes
                    ]

                    if any(has_corresp_opt):
                        v = (
                            "torch.optim."
                            + optimizers_classes[has_corresp_opt.index(True)].__name__
                        )
                    if k == "optimizer_class":
                        k = "-".join(k.split("_"))
                    cmd += f" --{k} {v}"
            # errfile = tmp_results_file.split(".")[0] + ".txt"
            # cmd += f" &> {errfile}"
            subprocess.run(cmd, shell=True)
            rfile_paths.append(tmp_results_file)
            new_r = pd.read_csv(tmp_results_file)
            for col in strat_res.columns:
                if col not in new_r:
                    new_r[col] = np.nan
            assert_dfs_equal(strat_res, new_r)
        centers_indices = [
            int(re.findall("(?<=Local )[0-9]{1}", e)[0])
            for e in r["Method"].unique()
            if len(re.findall("(?<=Local )[0-9]{1}", e)) > 0
        ]
        ncenters = max(centers_indices) + 1
        for i in range(ncenters):
            tmp_results_file = f"local{i}_seed{s}.csv"
            cmd = Path(flamby.__file__).parent / "benchmarks/fed_benchmark.py"
            cmd = "yes | python " + str(cmd)

            cmd += f" --seed {s} -cfp {cfp} -rfp {tmp_results_file} --nlocal {i}"
            cmd += " --single-centric-baseline Local"
            subprocess.run(cmd, shell=True)
            # errfile = tmp_results_file.split(".")[0] + ".txt"
            # cmd += f" &> {errfile}"
            local_from_all = r.loc[r["Method"] == f"Local {i}"]
            rfile_paths.append(tmp_results_file)
            new_r = pd.read_csv(tmp_results_file)
            for col in local_from_all.columns:
                if col not in new_r:
                    new_r[col] = np.nan
            assert_dfs_equal(local_from_all, new_r)

        tmp_results_file = f"pooled_seed{s}.csv"
        cmd = Path(flamby.__file__).parent / "benchmarks/fed_benchmark.py"
        cmd = "yes | python " + str(cmd)
        cmd += f" --seed {s} -cfp {cfp} -rfp {tmp_results_file}"
        cmd += " --single-centric-baseline Pooled"
        # errfile = tmp_results_file.split(".")[0] + ".txt"
        # cmd += f" &> {errfile}"
        subprocess.run(cmd, shell=True)
        pooled_from_all = r.loc[r["Method"] == "Pooled Training"]
        rfile_paths.append(tmp_results_file)
        new_r = pd.read_csv(tmp_results_file)
        for col in local_from_all.columns:
            if col not in new_r:
                new_r[col] = np.nan
        assert_dfs_equal(pooled_from_all, new_r)

    cleanup(rfile_paths)


def launch_all_xps_from_config(dataset_name, nrep=2):
    """Get the associated template config from the dataset and launch
    fed_benchmark with 5 seeds repeated nrep times.

    Parameters
    ----------
    dataset_nameme : str
        The name of the dataset
    nrep : int, optional
        The number of repetitions, by default 5

    Returns
    -------
    (list, list, str)
        The list of results pandas, filenames and the associated config file.
    """
    cfp = str(Path(flamby.__file__).parent / f"config_{dataset_name}.json")
    nseeds = 3
    seeds = range(42, 42 + nseeds)
    repetitions = []
    filenames = []
    for i in range(nrep):
        seeds_results = []
        for s in seeds:
            filename = f"{dataset_name}_seed{s}_rep{i}.csv"
            cmd = Path(flamby.__file__).parent / "benchmarks/fed_benchmark.py"
            cmd = "yes | python " + str(cmd)
            cmd += f" --seed {s} -cfp {cfp} -rfp {filename}"
            # errfile = filename.split(".")[0] + ".txt"
            # cmd += f" &> {errfile}"
            subprocess.run(cmd, shell=True)

            filenames.append(filename)
            seeds_results.append(pd.read_csv(filename))
        repetitions.append(seeds_results)
    return repetitions, filenames, cfp


def test_tcga():
    seeding_performance_assert("tcga_brca")
    compare_single_centric_and_strategy_vs_all("tcga_brca")


@pytest.mark.skip(reason="Need of downloading dataset")
def test_heart():
    seeding_performance_assert("heart_disease")
    compare_single_centric_and_strategy_vs_all("heart_disease")
