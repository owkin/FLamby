import itertools
import os

import numpy as np
import pandas as pd


def cleanup(files_list):
    [os.remove(file) for file in files_list]


def seeding_performance_assert(dataset_name, nrep=5):
    cfp = f"../../flamby/config_{dataset_name}.json"
    seeds = range(42, 47)
    repetitions = []
    filenames = []
    for i in range(nrep):
        seeds_results = []
        for s in seeds:
            filename = f"{dataset_name}_seed{s}_rep{i}.csv"
            os.system(
                f"python ../../flamby/benchmarks/fed_benchmark.py --seed {s} -cfp {cfp} -rfp {filename} --debug"
            )
            filenames.append(filename)
            seeds_results.append(pd.read_csv(filename))
        repetitions.append(seeds_results)
    for sidx in range(len(seeds)):
        all_dfs_for_seed = [rep[sidx] for rep in repetitions]
        paired_df = itertools.combinations(all_dfs_for_seed, 2)
        for pair in paired_df:
            df1 = pair[0].fillna("-9")
            df2 = pair[1].fillna("-9")[pair[0].columns]
            assert df1.drop(columns=["Metric"]).equals(df2.drop(columns=["Metric"]))
            assert np.allclose(df1["Metric"], df2["Metric"])

    cleanup(filenames)


def test_tcga():
    seeding_performance_assert("tcga_brca")


def test_heart():
    seeding_performance_assert("heart_disease")
