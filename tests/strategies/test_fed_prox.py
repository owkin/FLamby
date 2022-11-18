import copy
import shutil

import numpy as np
import pytest
import torch
import torch.utils.data as data
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader as dl
from torch.utils.data._utils.collate import default_collate
from torchvision import datasets
from torchvision.transforms import ToTensor

from flamby.datasets.fed_dummy_dataset import Baseline, BaselineLoss, FedDummyDataset
from flamby.strategies.fed_prox import FedProx
from flamby.utils import evaluate_model_on_tests


class NeuralNetwork(nn.Module):
    def __init__(self, lambda_dropout=0.05):
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


def cleanup():
    shutil.rmtree("./data")


@pytest.mark.parametrize("n_clients", [1, 2, 10])
def test_fed_prox_integration(n_clients):
    # tests if fed_prox is not failing on the MNIST dataset
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
    num_updates = 100
    nrounds = 50
    lr = 0.001
    mu = 0.1
    optimizer_class = torch.optim.Adam

    s = FedProx(train_dataloader, m, loss, optimizer_class, lr, num_updates, nrounds, mu)
    m = s.run()

    def accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred.argmax(axis=1))

    res = evaluate_model_on_tests(m[0], [test_dataloader], accuracy)

    print("\nAccuracy client 0:", res["client_test_0"])
    assert res["client_test_0"] > 0.95

    cleanup()


@pytest.mark.parametrize(
    "seed, lr, mu",
    [(42, 0.01, 0.0), (43, 0.001, 1.0), (44, 0.0001, 5.0), (45, 7e-5, 5e-4)],
)
def test_fed_prox_algorithm(seed, lr, mu):
    r"""FedProx should add an anchor term from the global model at each round.
    The implementation provided by the authors show that the wanted behavior
    is to update the weights to be - lr * grad + mu \cdot (var - global_var)

    Parameters
    ----------
    seed : int
        The seed to test.
    betas: tuple
        The betas associated with ADAM to control both momentums.
    epsilon: float
        The numerical precision parameter of ADAM in order to avoid dividing by 0.
    """

    num_updates = 10
    num_rounds = 1
    loss = BaselineLoss()
    torch.manual_seed(seed)

    m1 = Baseline().to(torch.double)
    m2 = copy.deepcopy(m1)

    def collate_fn_double(batch):
        outputs = default_collate(batch)
        return [o.to(torch.double) for o in outputs]

    # FedAdam
    torch.manual_seed(seed)
    training_dataloaders = [
        dl(
            FedDummyDataset(center=0, train=True, pooled=True),
            batch_size=32,
            shuffle=False,
            collate_fn=collate_fn_double,
        )
    ]
    weights_global_model = [p.detach().clone() for p in m1.parameters()]

    s = FedProx(
        training_dataloaders,
        m1,
        loss,
        torch.optim.SGD,
        lr,
        num_updates=num_updates,
        nrounds=num_rounds,
        log=False,
        mu=mu,
    )
    m1 = s.run()[0]
    weights_model_after_fedprox = [p.detach().numpy() for p in m1.parameters()]

    # Reference
    torch.manual_seed(seed)
    training_dataloaders = [
        dl(
            FedDummyDataset(center=0, train=True, pooled=True),
            batch_size=32,
            shuffle=False,
            collate_fn=collate_fn_double,
        )
    ]
    it = iter(training_dataloaders[0])
    opt = torch.optim.SGD(m2.parameters(), lr=lr)
    for i in range(num_updates):
        opt.zero_grad()
        X, y = next(it)
        y_pred = m2(X)
        ls = loss(y_pred, y)
        ls.backward()
        # We compare to an implementation that follows more closely
        # https://github.com/litian96/FedProx/blob/master/flearn/optimizer/pgd.py
        with torch.no_grad():
            for idx, (var, vstar) in enumerate(
                zip(m2.parameters(), weights_global_model)
            ):
                var.data -= lr * mu * (var.data - vstar)
        opt.step()

    weights_model_ref = [p.detach().numpy() for p in m2.parameters()]

    assert all(
        [
            np.allclose(w1, w2)
            for w1, w2 in zip(weights_model_ref, weights_model_after_fedprox)
        ]
    )
