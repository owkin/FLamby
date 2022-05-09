import copy

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader as dl
from torch.utils.data._utils.collate import default_collate

from flamby.datasets.fed_tcga_brca import Baseline, BaselineLoss, FedTcgaBrca
from flamby.strategies.fed_prox import FedProx


@pytest.mark.parametrize(
    "seed, lr, mu",
    [(42, 0.01, 0.0), (43, 0.001, 1.0), (44, 0.0001, 5.0), (45, 7e-5, 5e-4)],
)
def test_fed_prox(seed, lr, mu):
    """FedProx should add an anchor term from the global model at each round.
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
            FedTcgaBrca(center=0, train=True, pooled=True),
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
            FedTcgaBrca(center=0, train=True, pooled=True),
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
