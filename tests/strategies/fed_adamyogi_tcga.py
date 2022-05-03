import torch
from torch.utils.data import DataLoader as dl

from flamby.datasets.fed_tcga_brca import (
    BATCH_SIZE,
    LR,
    NUM_CLIENTS,
    Baseline,
    BaselineLoss,
    FedTcgaBrca,
    metric,
)
from flamby.datasets.fed_tcga_brca.common import get_nb_max_rounds
from flamby.strategies.fed_adam_yogi import FedAdamYogi
from flamby.utils import evaluate_model_on_tests


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def fedadamyogi_TCGA():

    results = []

    for seed in range(5):

        torch.manual_seed(seed)

        training_dls = [
            dl(
                FedTcgaBrca(center=i, train=True, pooled=False),
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=4,
            )
            for i in range(NUM_CLIENTS)
        ]
        test_dls = [
            dl(
                FedTcgaBrca(center=i, train=False, pooled=False),
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=4,
            )
            for i in range(NUM_CLIENTS)
        ]
        loss = BaselineLoss()
        m = Baseline()
        NUM_UPDATES = 1
        nrounds = get_nb_max_rounds(NUM_UPDATES)
        optimizer_class = torch.optim.SGD
        s = FedAdamYogi(
            training_dls,
            m,
            loss,
            optimizer_class,
            LR,
            NUM_UPDATES,
            nrounds,
            log=False,
            tau=1e-3,
            server_learning_rate=1e-2,
            yogi=True,
        )
        m = s.run()
        results.append(evaluate_model_on_tests(m[0], test_dls, metric))

    print(results)
    print(dict_mean(results))


if __name__ == "__main__":

    fedadamyogi_TCGA()
