import copy
import pdb
from itertools import product

# Plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader as dl

from flamby.datasets.fed_ixi import (
    BATCH_SIZE,
    LR,
    NUM_CLIENTS,
    Baseline,
    BaselineLoss,
    FedIXITiny,
    get_nb_max_rounds,
    metric,
)
from flamby.strategies import FedAvg
from flamby.utils import evaluate_model_on_tests

sns.set_theme(style="darkgrid")

n_repetitions = 5
num_updates = 100
nrounds = get_nb_max_rounds(num_updates)


bloss = BaselineLoss()
# We init the strategy parameters to the following default ones

args = {
    "loss": bloss,
    "optimizer_class": torch.optim.SGD,
    "learning_rate": LR,
    "num_updates": num_updates,
    "nrounds": nrounds,
}

epsilons = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
deltas = [10 ** (-i) for i in range(2, 6)]
seeds = np.arange(42, 42 + n_repetitions).tolist()

test_dls = [
    dl(
        FedIXITiny(center=i, train=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=10,
    )
    for i in range(NUM_CLIENTS)
]

results_all_reps = []
for s in seeds:
    global_init = Baseline()
    torch.manual_seed(s)
    training_dls = [
        dl(
            FedIXITiny(center=i, train=True),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )
        for i in range(NUM_CLIENTS)
    ]
    args["training_dataloaders"] = training_dls
    for e, d in product(epsilons, deltas):
        current_args = copy.deepcopy(args)
        current_args["model"] = copy.deepcopy(global_init)
        current_args["dp_target_epsilon"] = e
        current_args["dp_target_delta"] = d
        # We run FedAvg
        s = FedAvg(**current_args, log=False)
        cm = s.run()[0]
        mean_perf = np.array(
            [v for _, v in evaluate_model_on_tests(cm, test_dls, metric).items()]
        ).mean()
        results_all_reps.append({"perf": mean_perf, "e": e, "d": d, "seed": s})

results = pd.DataFrame.from_dict(results_all_reps)

pdb.set_trace()

results.to_csv("results_ixi_dp.csv", index=False)


results = pd.read_csv("results_ixi_dp.csv")


for d in deltas:
    cdf = results.loc[results["delta"] == d]
    sns.scatterplot(data=cdf, x="epsilons", y="perf", label=f"delta={d}")
plt.legend()
plt.waitforbuttonpress()
