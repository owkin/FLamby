import pytest
import torch
import torch.utils.data as data
from torch import nn
from torch.utils.data import DataLoader as dl
from torchvision import datasets
from torchvision.transforms import ToTensor

from flamby.datasets.fed_isic2019 import (
    BATCH_SIZE,
    LR,
    NUM_CLIENTS,
    Baseline,
    BaselineLoss,
    FedIsic2019,
    metric,
)
from flamby.datasets.fed_isic2019.common import get_nb_max_rounds
from flamby.strategies.fed_avg import FedAvgTorch
from flamby.utils import evaluate_model_on_tests


@pytest.mark.parametrize("n_clients", [1, 2, 3, 4])
def test_fed_avg(n_clients):
    # tests if fed_avg is not failing on the MNIST dataset
    # with different number of clients
    # get the data
    training_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    # split to n_clients
    splits = [int(len(training_data) / n_clients)] * n_clients
    splits[-1] = splits[-1] + len(training_data) % n_clients
    training_data = data.random_split(training_data, splits)

    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )

    train_dataloader = [
        dl(train_data, batch_size=64, shuffle=True) for train_data in training_data
    ]
    test_dataloader = dl(test_data, batch_size=64, shuffle=True)
    loss = nn.CrossEntropyLoss()
    m = NeuralNetwork()
    nrounds = 20
    lr = 0.05
    optimizer = torch.optim.SGD(m.parameters(), lr=lr)
    s = FedAvgTorch(train_dataloader, m, loss, nrounds, optimizer=optimizer)
    m = s.run()
    print(
        evaluate_model_on_tests(m[0], [test_dataloader], metric)
    )  # TODO: this won't be printed, add a test


class NeuralNetwork(nn.Module):
    def __init__(self, lambda_dropout=0.05):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        n_hidden = 8
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.Dropout(lambda_dropout),
            nn.ReLU(),
            nn.Linear(n_hidden, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def test_fedavg_Isic():
    # tests if fedavg is not failing with ISIC
    training_dls = [
        dl(
            FedIsic2019(train=True, center=i),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=10,
        )
        for i in range(NUM_CLIENTS)
    ]
    test_dls = [
        dl(
            FedIsic2019(train=False, center=i),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=10,
        )
        for i in range(NUM_CLIENTS)
    ]
    loss = BaselineLoss()
    m = Baseline()
    NUM_UPDATES = 50
    nrounds = get_nb_max_rounds(NUM_UPDATES)
    optimizer = torch.optim.Adam(m.base_model.parameters(), lr=LR)
    s = FedAvgTorch(training_dls, m, loss, nrounds, optimizer=optimizer)
    m = s.run()
    print(evaluate_model_on_tests(m, test_dls, metric))
