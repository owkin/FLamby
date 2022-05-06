import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader as dl

from flamby.datasets.fed_tcga_brca import (
    BATCH_SIZE,
    Baseline,
    BaselineLoss,
    FedTcgaBrca,
    metric,
)
from flamby.strategies.fed_opt import FedAdam
from flamby.utils import evaluate_model_on_tests


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def test_fed_adam():
    lossfunc = BaselineLoss()
    results0 = []
    results1 = []
    for seed in range(3):
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_dataset = FedTcgaBrca(train=True, pooled=True)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
        )
        test_dataset = FedTcgaBrca(train=False, pooled=True)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
        )
        dataloaders = {"train": train_dataloader, "test": test_dataloader}

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = Baseline()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

        results0.append(evaluate_model_on_tests(model, [test_dataloader], metric))

        for idx, (X, y) in enumerate(dataloaders["train"]):
            if idx == 0:
                X = X.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                outputs = model(X)
                loss = lossfunc(outputs, y)
                loss.backward()
                optimizer.step()
                scheduler.step()

        dict_cindex = evaluate_model_on_tests(model, [dataloaders["test"]], metric)
        results1.append(dict_cindex)

    print("Before training")
    print("Test C-index ", results0)
    print("Average test C-index ", dict_mean(results0))
    print("After training")
    print("Test C-index ", results1)
    print("Average test C-index ", dict_mean(results1))

    results2 = []
    results3 = []
    for seed in range(3):
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
        results2.append(evaluate_model_on_tests(m, test_dls, metric))

        NUM_UPDATES = 1
        nrounds = 1

        optimizer_class = torch.optim.SGD
        LR = 1e0
        SLR = (1e-1) * np.sqrt((1.0 - 0.999)) / (1.0 - 0.9)
        TAU = (1e-8) * ((1 - 0.999) ** 0.5)

        s = FedAdam(
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
        )
        m = s.run()
        results3.append(evaluate_model_on_tests(m[0], test_dls, metric))

    print("Test metric before training")
    print(results2)
    print(dict_mean(results2))
    print("Test metric after training")
    print(results3)
    print(dict_mean(results3))
