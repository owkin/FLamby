import copy

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader as dl
from torch.utils.data._utils.collate import default_collate

from flamby.datasets.fed_dummy_dataset import Baseline, BaselineLoss, FedDummyDataset
from flamby.strategies.fed_opt import FedAdam


@pytest.mark.parametrize(
    "seed, betas, epsilon",
    [(42, (0.5, 0.8), 1.0), (43, (0.9, 0.999), 1e-8), (44, (0.5, 0.5), 1.0e-4)],
)
def test_fed_adam(seed, betas, epsilon):
    """FedAdam is not entirely the same as Adam in the 1 client only case.
    Notably there is no bias correction in FedAdam.
    However by tweaking the server learning rate of FedAdam and under
    certain conditions one can have the equality between both.
    We need to use doubles as floats use cause frequent rounding errors,
    making tests fail.
    - client learning rate of 1. (LR)
    - sever learning rate of {\sqrt{1. - \beta_{2}}}{1. - \beta_{1}}
    - tau = epsilon \cdot \sqrt{1. - \beta_{2}}
    - first iteration of both algorithms and num_updates = 1

    Parameters
    ----------
    seed : int
        The seed to test.
    betas: tuple
        The betas associated with ADAM to control both momentums.
    epsilon: float
        The numerical precision parameter of ADAM in order to avoid dividing by 0.
    """

    LR = 1.0
    slr = np.sqrt((1.0 - betas[1])) / (1.0 - betas[0])
    num_updates = 1
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

    s = FedAdam(
        training_dataloaders,
        m1,
        loss,
        torch.optim.SGD,
        LR,
        num_updates=num_updates,
        nrounds=num_rounds,
        log=False,
        tau=epsilon * np.sqrt((1.0 - betas[1])),
        server_learning_rate=slr,
        beta1=betas[0],
        beta2=betas[1],
    )
    m1 = s.run()[0]
    weights_model_from_strat = [p.detach().numpy() for p in m1.parameters()]

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

    adam_optim = torch.optim.Adam(m2.parameters(), lr=LR, betas=betas, eps=epsilon)

    for i in range(num_updates):
        adam_optim.zero_grad()
        X, y = next(it)
        y_pred = m2(X)
        ls = loss(y_pred, y)
        ls.backward()
        adam_optim.step()

    weights_model_from_adam = [p.detach().numpy() for p in m2.parameters()]

    assert all(
        [
            np.allclose(w1, w2)
            for w1, w2 in zip(weights_model_from_strat, weights_model_from_adam)
        ]
    )
