import copy
import pdb
from itertools import product

# Plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader as dl
from tqdm import tqdm

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

torch.multiprocessing.set_sharing_strategy("file_system")

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
    "nrounds": 4,  # nrounds
}

epsilons = [0.1, 1.0, 5.0, 10.0, 50.0][::-1]
deltas = [10 ** (-i) for i in range(2, 6)]
seeds = np.arange(42, 42 + n_repetitions).tolist()

test_dls = [
    dl(
        FedIXITiny(center=i, train=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    for i in range(NUM_CLIENTS)
]

results_all_reps = []
edelta_list = list(product(epsilons, deltas))
baseline_perfs = []
for se in seeds:
    global_init = Baseline()
    torch.manual_seed(se)
    training_dls = [
        dl(
            FedIXITiny(center=i, train=True),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8,
        )
        for i in range(NUM_CLIENTS)
    ]
    args["training_dataloaders"] = training_dls
    current_args = copy.deepcopy(args)
    current_args["model"] = copy.deepcopy(global_init)

    # We run FedAvg wo DP
    s = FedAvg(**current_args, log=False)
    cm = s.run()[0]
    mean_perf = np.array(
        [v for _, v in evaluate_model_on_tests(cm, test_dls, metric).items()]
    ).mean()
    print(f"Mean performance without DP, Perf={mean_perf}")
    baseline_perfs.append(mean_perf)

    for e, d in tqdm(edelta_list):
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
        print(f"Mean performance eps={e}, delta={d}, Perf={mean_perf}")
        # mean_perf = float(np.random.uniform(0, 1.))
        results_all_reps.append({"perf": mean_perf, "e": e, "d": d, "seed": se})

results = pd.DataFrame.from_dict(results_all_reps)

# pdb.set_trace()

results.to_csv("results_ixi_dp.csv", index=False)


results = pd.read_csv("results_ixi_dp.csv")

linestyle_tuple = [
    ("loosely dotted", (0, (1, 10))),
    ("dotted", (0, (1, 1))),
    ("densely dotted", (0, (1, 1))),
    ("loosely dashed", (0, (5, 10))),
    ("dashed", (0, (5, 5))),
    ("densely dashed", (0, (5, 1))),
    ("loosely dashdotted", (0, (3, 10, 1, 10))),
    ("dashdotted", (0, (3, 5, 1, 5))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
]

for i, d in enumerate(deltas):
    cdf = results.loc[results["d"] == d]
    fig = sns.lineplot(
        data=cdf, x="e", y="perf", label=f"delta={d}", linestyle=linestyle_tuple[i][1]
    )
    fig.set(xscale="log")
fig.axhline(np.array(baseline_perfs).mean(), color="black")
plt.legend()
plt.xlabel("epsilon")
plt.ylabel("Perf")
plt.savefig("trial.png")
