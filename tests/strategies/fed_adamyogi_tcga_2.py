import numpy as np
import torch
from torch.utils.data import DataLoader as dl

from flamby.datasets.fed_tcga_brca import (
    BATCH_SIZE,
    Baseline,
    BaselineLoss,
    FedTcgaBrca,
    metric,
)
from flamby.strategies.fed_adam_yogi import FedAdamYogi
from flamby.utils import evaluate_model_on_tests


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def fedadamyogi_TCGA():

    results0 = []
    results = []

    for seed in range(5):

        torch.manual_seed(seed)
        np.random.seed(seed)
        NUM_CLIENTS = 1

        training_dls = [
            dl(
                FedTcgaBrca(train=True, pooled=True),
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=4,
            )
            for i in range(NUM_CLIENTS)
        ]
        test_dls = [
            dl(
                FedTcgaBrca(train=False, pooled=True),
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=4,
            )
            for i in range(NUM_CLIENTS)
        ]
        loss = BaselineLoss()
        m = Baseline()
        results0.append(evaluate_model_on_tests(m, test_dls, metric))

        NUM_UPDATES = 1
        # nrounds = get_nb_max_rounds(NUM_UPDATES)
        nrounds = (866 // BATCH_SIZE) * 10 // NUM_UPDATES

        optimizer_class = torch.optim.SGD
        LR = 1e-1  # 1.0
        SLR = 1e-1  # 1e-1
        TAU = 1e-3  # 1e-8
        s = FedAdamYogi(
            training_dls,
            m,
            loss,
            optimizer_class,
            LR,
            NUM_UPDATES,
            nrounds,
            log=False,
            tau=TAU,
            server_learning_rate=SLR,
            yogi=True,
        )
        m = s.run()
        results.append(evaluate_model_on_tests(m[0], test_dls, metric))

    print("Test metric before training")
    print(results0)
    print(dict_mean(results0))
    print("Test metric after training")
    print(results)
    print(dict_mean(results))


if __name__ == "__main__":

    fedadamyogi_TCGA()
