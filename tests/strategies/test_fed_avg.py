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
from flamby.strategies.fed_avg import FedAvg
from flamby.utils import evaluate_model_on_tests


@pytest.mark.parametrize("n_clients", [1, 2, 10])
def test_fed_avg(n_clients):
    # tests if fed_avg is not failing on the MNIST dataset
    # with different number of clients
    # get the data
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

    train_dataloader = [
        dl(train_data, batch_size=100, shuffle=True) for train_data in training_data
    ]
    test_dataloader = dl(test_data, batch_size=100, shuffle=False)
    loss = nn.CrossEntropyLoss()
    m = NeuralNetwork()
    nrounds = 50
    lr = 0.001
    optimizer_class = torch.optim.Adam

    s = FedAvg(train_dataloader, m, loss, optimizer_class, lr, 100, nrounds)
    m = s.run()

    res = evaluate_model_on_tests(m[0], [test_dataloader], metric)

    assert res["client_test_0"] > 0.8


class NeuralNetwork(nn.Module):
    def __init__(self, lambda_dropout=0.05):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        n_hidden = 128
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            # nn.Dropout(lambda_dropout),
            nn.ReLU(),
            nn.Linear(n_hidden, 10),
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
            num_workers=0,
        )
        for i in range(NUM_CLIENTS)
    ]
    test_dls = [
        dl(
            FedIsic2019(train=False, center=i),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
        for i in range(NUM_CLIENTS)
    ]
    loss = BaselineLoss()
    m = Baseline()
    NUM_UPDATES = 2
    nrounds = get_nb_max_rounds(NUM_UPDATES)
    optimizer_class = torch.optim.Adam
    breakpoint()
    s = FedAvg(training_dls, m, loss, optimizer_class, LR, NUM_UPDATES, nrounds)
    m = s.run()
    print(evaluate_model_on_tests(m, test_dls, metric))
