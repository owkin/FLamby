import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from flamby.strategies.fed_avg import FedAvg
from flamby.strategies.fed_opt import FedAdam, FedOpt


class ToyDataset(Dataset):
    """A small deterministic dataset, to keep the test independent of torchvision."""

    def __init__(self, seed, n=32):
        generator = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 3, generator=generator)
        self.y = (self.x.sum(dim=1, keepdim=True) > 0).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def _loaders():
    return [
        DataLoader(ToyDataset(0), batch_size=8, shuffle=False),
        DataLoader(ToyDataset(1), batch_size=8, shuffle=False),
    ]


def _local_deltas(model, loaders, num_updates):
    """Recompute the local updates by hand, following the FedAvg definition."""
    deltas = []
    for loader in loaders:
        local = copy.deepcopy(model)
        optimizer = torch.optim.SGD(local.parameters(), lr=0.1)
        loss_fn = nn.BCEWithLogitsLoss()
        for batch_index, (x, y) in enumerate(loader):
            if batch_index >= num_updates:
                break
            optimizer.zero_grad()
            loss_fn(local(x), y).backward()
            optimizer.step()
        deltas.append(
            [
                new.detach() - old.detach()
                for new, old in zip(local.parameters(), model.parameters())
            ]
        )
    return deltas


def test_fed_opt_inherits_the_aggregation_from_fed_avg():
    assert FedOpt.calc_aggregated_delta_weights is FedAvg.calc_aggregated_delta_weights


def test_perform_round_is_the_weighted_average_of_local_updates():
    torch.manual_seed(0)
    model = nn.Linear(3, 1, bias=False)
    loaders = _loaders()
    num_updates = 2

    strategy = FedAvg(
        loaders,
        copy.deepcopy(model),
        nn.BCEWithLogitsLoss(),
        torch.optim.SGD,
        learning_rate=0.1,
        num_updates=num_updates,
        nrounds=1,
        seed=0,
    )

    deltas = _local_deltas(model, loaders, num_updates)
    sizes = [len(loader.dataset) for loader in loaders]
    total = sum(sizes)
    expected = [
        sum(delta[i] * size for delta, size in zip(deltas, sizes)) / total
        for i in range(len(deltas[0]))
    ]

    aggregated = strategy.calc_aggregated_delta_weights()
    for got, want in zip(aggregated, expected):
        assert torch.allclose(torch.as_tensor(got), want, atol=1e-6)

    # A round recomputes the local updates, so it needs its own strategy object.
    rounding = FedAvg(
        _loaders(),
        copy.deepcopy(model),
        nn.BCEWithLogitsLoss(),
        torch.optim.SGD,
        learning_rate=0.1,
        num_updates=num_updates,
        nrounds=1,
        seed=0,
    )
    rounding.perform_round()
    updated = [p.detach() for p in rounding.models_list[0].model.parameters()]
    for got, start, delta in zip(updated, model.parameters(), expected):
        assert torch.allclose(got, start.detach() + delta, atol=1e-6)


@pytest.mark.parametrize("algorithm", [FedAdam])
def test_fed_opt_variants_still_run(algorithm):
    torch.manual_seed(0)
    strategy = algorithm(
        _loaders(),
        nn.Linear(3, 1, bias=False),
        nn.BCEWithLogitsLoss(),
        torch.optim.SGD,
        learning_rate=0.1,
        num_updates=1,
        nrounds=1,
        seed=0,
    )
    strategy.perform_round()
    assert all(
        torch.isfinite(p).all() for p in strategy.models_list[0].model.parameters()
    )
