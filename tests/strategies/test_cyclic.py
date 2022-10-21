import shutil

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from flamby.datasets.fed_camelyon16 import (
    BATCH_SIZE,
    LR,
    NUM_CLIENTS,
    Baseline,
    BaselineLoss,
    FedCamelyon16,
    collate_fn,
    get_nb_max_rounds,
    metric,
)
from flamby.strategies.cyclic import Cyclic
from flamby.utils import evaluate_model_on_tests

NUM_LOCAL_EPOCHS = 2
SEED = 42
LOG_PERIOD = 10


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        n_hidden = 128
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred.argmax(axis=1))


def cleanup():
    shutil.rmtree("./data")


@pytest.mark.parametrize("n_clients", [1, 2, 10])
def test_cyclic(n_clients):
    # tests if fed_avg is not failing on the MNIST dataset
    # with different number of clients
    num_updates = 100
    nrounds = 50
    lr = 0.001
    batch_size = 100

    training_data = datasets.MNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    # split to n_clients
    splits = [int(len(training_data) / n_clients)] * n_clients
    splits[-1] = splits[-1] + len(training_data) % n_clients
    training_data = data.random_split(training_data, splits)

    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )

    train_dataloaders = [
        DataLoader(train_data, batch_size=batch_size, shuffle=True)
        for train_data in training_data
    ]
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    loss = nn.CrossEntropyLoss()
    model = NeuralNetwork()

    optimizer_class = torch.optim.Adam

    s = Cyclic(
        training_dataloaders=train_dataloaders,
        model=model,
        loss=loss,
        optimizer_class=optimizer_class,
        learning_rate=lr,
        num_updates=num_updates,
        nrounds=nrounds,
        log=True,
        log_period=LOG_PERIOD,
        deterministic_cycle=False,
        rng=np.random.default_rng(SEED),
    )

    print("\nStarting training ...")
    m = s.run()

    res = evaluate_model_on_tests(m[0], [test_dataloader], accuracy)

    print("\nAccuracy client 0:", res["client_test_0"])
    assert res["client_test_0"] > 0.90

    cleanup()


@pytest.mark.skip(reason="Need of downloading dataset")
def test_cyclic_camelyon():
    training_dls = [
        DataLoader(
            FedCamelyon16(train=True, center=i),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=10,
            collate_fn=collate_fn,
        )
        for i in range(NUM_CLIENTS)
    ]

    test_dls = [
        DataLoader(
            FedCamelyon16(train=False, center=i),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=10,
            collate_fn=collate_fn,
        )
        for i in range(NUM_CLIENTS)
    ]

    loss = BaselineLoss()
    m = Baseline()

    nrounds = get_nb_max_rounds(NUM_LOCAL_EPOCHS)

    s = Cyclic(
        training_dataloaders=training_dls,
        model=m,
        loss=loss,
        optimizer_class=optim.SGD,
        learning_rate=LR,
        num_updates=NUM_LOCAL_EPOCHS,
        nrounds=nrounds,
        log=True,
        log_period=LOG_PERIOD,
        deterministic_cycle=True,
        rng=np.random.default_rng(SEED),
    )

    print("\nStarting training ...")

    m = s.run()

    print(evaluate_model_on_tests(m, test_dls, metric))
