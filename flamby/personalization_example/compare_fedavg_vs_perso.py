import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader as dl
from tqdm import tqdm

# from flamby.datasets.fed_camelyon16 import LR as fedcam16_lr
from flamby.datasets.fed_camelyon16 import BATCH_SIZE as fedcam16_batch_size
from flamby.datasets.fed_camelyon16 import NUM_CLIENTS as fedcam16_num_clients
from flamby.datasets.fed_camelyon16 import Baseline as FedCam16Baseline
from flamby.datasets.fed_camelyon16 import BaselineLoss as FedCam16BaselineLoss
from flamby.datasets.fed_camelyon16 import FedCamelyon16, collate_fn
from flamby.datasets.fed_camelyon16 import (
    get_nb_max_rounds as fedcam16_get_nb_max_rounds,
)
from flamby.datasets.fed_camelyon16 import metric as fedcam16_metric

# from flamby.datasets.fed_heart_disease import LR as fedheart_lr
from flamby.datasets.fed_heart_disease import BATCH_SIZE as fedheart_batch_size
from flamby.datasets.fed_heart_disease import NUM_CLIENTS as fedheart_num_clients
from flamby.datasets.fed_heart_disease import Baseline as FedHeartBaseline
from flamby.datasets.fed_heart_disease import BaselineLoss as FedHeartBaselineLoss
from flamby.datasets.fed_heart_disease import FedHeartDisease
from flamby.datasets.fed_heart_disease import (
    get_nb_max_rounds as fedheart_get_nb_max_rounds,
)
from flamby.datasets.fed_heart_disease import metric as fedheart_metric

# from flamby.datasets.fed_isic2019 import LR as fedisic19_lr
from flamby.datasets.fed_isic2019 import BATCH_SIZE as fedisic19_batch_size
from flamby.datasets.fed_isic2019 import NUM_CLIENTS as fedisic19_num_clients
from flamby.datasets.fed_isic2019 import Baseline as FedIsic2019Baseline
from flamby.datasets.fed_isic2019 import BaselineLoss as FedIsic2019BBaselineLoss
from flamby.datasets.fed_isic2019 import FedIsic2019
from flamby.datasets.fed_isic2019 import (
    get_nb_max_rounds as fedisic19_get_nb_max_rounds,
)
from flamby.datasets.fed_isic2019 import metric as fedisic19_metric

# strategies
from flamby.strategies import FedAvg, FedAvgFineTuning

# evaluation
from flamby.utils import evaluate_model_on_tests

torch.multiprocessing.set_sharing_strategy("file_system")

num_updates = 100
n_repetitions = 5

generic_args = {
    "loss": {
        "Fed-Heart-Disease": FedHeartBaselineLoss(),
        "Fed-Camelyon16": FedCam16BaselineLoss(),
        "Fed-ISIC2019": FedIsic2019BBaselineLoss(),
    },
    "optimizer_class": torch.optim.SGD,
    "learning_rate": {
        "Fed-Heart-Disease": 0.00316227,
        "Fed-Camelyon16": 0.316227,
        "Fed-ISIC2019": 0.01,
    },
    "num_updates": num_updates,
    "nrounds": {
        "Fed-Heart-Disease": fedheart_get_nb_max_rounds(100),
        "Fed-Camelyon16": fedcam16_get_nb_max_rounds(100),
        "Fed-ISIC2019": fedisic19_get_nb_max_rounds(100),
    },
}
models_architectures = {
    "Fed-Heart-Disease": FedHeartBaseline,
    "Fed-Camelyon16": FedCam16Baseline,
    "Fed-ISIC2019": FedIsic2019Baseline,
}
datasets_classes = {
    "Fed-Heart-Disease": FedHeartDisease,
    "Fed-Camelyon16": FedCamelyon16,
    "Fed-ISIC2019": FedIsic2019,
}
datasets_batch_sizes = {
    "Fed-Heart-Disease": fedheart_batch_size,
    "Fed-Camelyon16": fedcam16_batch_size,
    "Fed-ISIC2019": fedisic19_batch_size,
}
datasets_num_clients = {
    "Fed-Heart-Disease": fedheart_num_clients,
    "Fed-Camelyon16": fedcam16_num_clients,
    "Fed-ISIC2019": fedisic19_num_clients,
}
datasets_metrics = {
    "Fed-Heart-Disease": fedheart_metric,
    "Fed-Camelyon16": fedcam16_metric,
    "Fed-ISIC2019": fedisic19_metric,
}

datasets_names = list(models_architectures.keys())
seeds = np.arange(42, 42 + n_repetitions).tolist()

results_all_reps = []

for dn in tqdm(datasets_names):
    for se in tqdm(seeds):
        # We set model and dataloaders to be the same for each rep
        global_init = models_architectures[dn]()
        torch.manual_seed(se)
        if dn == "Fed-Camelyon16":
            training_dls = [
                dl(
                    datasets_classes[dn](center=i, train=True),
                    batch_size=datasets_batch_sizes[dn],
                    shuffle=True,
                    num_workers=0,
                    collate_fn=collate_fn,
                )
                for i in range(datasets_num_clients[dn])
            ]
            test_dls = [
                dl(
                    datasets_classes[dn](center=i, train=False),
                    batch_size=datasets_batch_sizes[dn],
                    shuffle=True,
                    num_workers=0,
                    collate_fn=collate_fn,
                )
                for i in range(datasets_num_clients[dn])
            ]
        else:
            training_dls = [
                dl(
                    datasets_classes[dn](center=i, train=True),
                    batch_size=datasets_batch_sizes[dn],
                    shuffle=True,
                    num_workers=0,
                )
                for i in range(datasets_num_clients[dn])
            ]
            test_dls = [
                dl(
                    datasets_classes[dn](center=i, train=False),
                    batch_size=datasets_batch_sizes[dn],
                    shuffle=True,
                    num_workers=0,
                )
                for i in range(datasets_num_clients[dn])
            ]

        args = {
            "loss": generic_args["loss"][dn],
            "optimizer_class": torch.optim.SGD,
            "learning_rate": generic_args["learning_rate"][dn],
            "num_updates": num_updates,
            "nrounds": generic_args["nrounds"][dn],
        }
        args["training_dataloaders"] = training_dls
        current_args = copy.deepcopy(args)
        current_args["model"] = copy.deepcopy(global_init)

        # We run FedAvg wo DP
        s = FedAvg(**current_args, log=False)
        cm = s.run()[0]
        mean_perf = np.array(
            [
                v
                for _, v in evaluate_model_on_tests(
                    cm, test_dls, datasets_metrics[dn]
                ).items()
            ]
        ).mean()
        current_row = {"perf": mean_perf, "dataset": dn, "finetune": False, "s": se}
        print(current_row)

        results_all_reps.append(current_row)

        # We run FedAvg with fine-tuning
        current_args = copy.deepcopy(args)
        current_args["model"] = copy.deepcopy(global_init)
        current_args["num_fine_tuning_steps"] = 100

        s = FedAvgFineTuning(**current_args, log=False)
        cms = s.run()
        # We test each personalized model on its corresponding test set
        perfs = []
        for i in range(datasets_num_clients[dn]):
            perf_dict = evaluate_model_on_tests(cms[i], test_dls, datasets_metrics[dn])
            perfs.append(perf_dict[list(perf_dict.keys())[i]])
        mean_perf = np.array(perfs).mean()
        current_row = {"perf": mean_perf, "dataset": dn, "finetune": True, "s": se}
        print(current_row)

        results_all_reps.append(current_row)


results = pd.DataFrame.from_dict(results_all_reps)
results.to_csv("results_perso_vs_normal.csv", index=False)
