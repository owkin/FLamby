import copy
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader as dl
from tqdm import tqdm

from flamby.datasets.fed_heart_disease import (
    BATCH_SIZE,
    LR,
    NUM_CLIENTS,
    Baseline,
    BaselineLoss,
    FedHeartDisease,
    get_nb_max_rounds,
    metric,
)
from flamby.strategies import FedAvg
from flamby.utils import evaluate_model_on_tests

torch.multiprocessing.set_sharing_strategy("file_system")


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

epsilons = [0.1, 1.0, 5.0, 10.0, 50.0][::-1]
deltas = [10 ** (-i) for i in range(1, 5)]
START_SEED = 42
seeds = np.arange(START_SEED, START_SEED + n_repetitions).tolist()

test_dls = [
    dl(
        FedHeartDisease(center=i, train=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        # collate_fn=collate_fn,
    )
    for i in range(NUM_CLIENTS)
]

results_all_reps = []
edelta_list = list(product(epsilons, deltas))
for se in seeds:
    # We set model and dataloaders to be the same for each rep
    global_init = Baseline()
    torch.manual_seed(se)
    training_dls = [
        dl(
            FedHeartDisease(center=i, train=True),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            # collate_fn=collate_fn,
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
    results_all_reps.append({"perf": mean_perf, "e": None, "d": None, "seed": se})

    for e, d in tqdm(edelta_list):
        current_args = copy.deepcopy(args)
        current_args["model"] = copy.deepcopy(global_init)
        current_args["dp_target_epsilon"] = e
        current_args["dp_target_delta"] = d
        current_args["dp_max_grad_norm"] = 1.1
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
results.to_csv("results_fed_heart_disease.csv", index=False)
