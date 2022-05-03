import torch
from torch.utils.data import DataLoader as dl

from flamby.datasets.fed_tcga_brca import (
    BATCH_SIZE,
    LR,
    Baseline,
    BaselineLoss,
    FedTcgaBrca,
    metric,
)
from flamby.datasets.fed_tcga_brca.common import get_nb_max_rounds
from flamby.strategies.fed_avg import FedAvg
from flamby.utils import evaluate_model_on_tests


def fedavg_TCGA():

    torch.manual_seed(0)

    training_dls = [
        dl(
            FedTcgaBrca(train=True, pooled=True),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
        )
        for i in range(1)
    ]
    test_dls = [
        dl(
            FedTcgaBrca(train=False, pooled=True),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
        )
        for i in range(1)
    ]

    loss = BaselineLoss()
    m = Baseline()
    NUM_UPDATES = 1
    nrounds = get_nb_max_rounds(NUM_UPDATES)
    optimizer_class = torch.optim.Adam
    s = FedAvg(
        training_dls, m, loss, optimizer_class, LR, NUM_UPDATES, nrounds, log=False
    )
    m = s.run()
    print(evaluate_model_on_tests(m[0], test_dls, metric))


if __name__ == "__main__":

    fedavg_TCGA()
