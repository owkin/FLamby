import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate

from flamby.datasets.fed_tcga_brca import BATCH_SIZE, LR, Baseline, BaselineLoss
from flamby.datasets.fed_tcga_brca import FedTcgaBrca as FedDataset
from flamby.datasets.fed_tcga_brca import Optimizer


class OldBaselineLoss(nn.Module):
    """
    Old version of the baseline loss
    """

    """Compute Cox loss given model output and ground truth (E, T)
    Parameters
    ----------
    scores: torch.Tensor, float tensor of dimension (n_samples, 1), typically
        the model output.
    truth: torch.Tensor, float tensor of dimension (n_samples, 2) containing
        ground truth event occurrences 'E' and times 'T'.
    Returns
    -------
    torch.Tensor of dimension (1, ) giving mean of Cox loss.
    """

    def __init__(self):
        super(OldBaselineLoss, self).__init__()

    def forward(self, scores, truth):
        # The Cox loss calc expects events to be reverse sorted in time
        a = torch.stack((torch.squeeze(scores, dim=1), truth[:, 0], truth[:, 1]), dim=1)
        a = torch.stack(sorted(a, key=lambda a: -a[2]))
        scores = a[:, 0]
        events = a[:, 1]
        scores_ = scores - scores.max()
        loss = -(scores_ - torch.log(torch.exp(scores_).cumsum(0))) * events
        return loss.mean()


def test_cox_loss():
    # start by checking the two losses behave the same
    # Instantiation of local train set (and data loader)),
    # baseline loss function, baseline model, default optimizer
    def collate_fn_double(batch):
        outputs = default_collate(batch)
        return [o.to(torch.double) for o in outputs]

    train_dataset = FedDataset(center=0, train=True, pooled=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_double,
    )
    lossfunc = BaselineLoss()
    lossfunc_old = OldBaselineLoss()
    model = Baseline().to(torch.double)
    model_old = copy.deepcopy(model)
    optimizer = Optimizer(model.parameters(), lr=LR)
    optimizer_old = Optimizer(model_old.parameters(), lr=LR)

    # guarantee that we get exactly the same behavior on one epoch between the two losses

    for idx, (X, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(X)
        loss = lossfunc(outputs, y)
        loss.backward()
        optimizer.step()

        optimizer_old.zero_grad()
        outputs_old = model_old(X)
        loss_old = lossfunc_old(outputs_old, y)
        loss_old.backward()
        optimizer_old.step()

        weights = [p.data.numpy() for p in model.parameters()]
        weights_old = [p.data.numpy() for p in model_old.parameters()]
        assert all(
            [np.allclose(w1, w2, atol=1e-6) for w1, w2 in zip(weights, weights_old)]
            + [abs(loss_old.item() - loss.item()) < 1e-6]
        )

    # Example where the new loss works but not the old one (for this, need to be float32)
    # the old one return an infinite value
    scores = torch.tensor([-400, 0]).view((2, 1)).to(torch.float)
    truth = torch.tensor([[1, 10], [0, 1]]).to(torch.float)
    old_value = lossfunc_old.forward(scores, truth).item()
    new_value = lossfunc.forward(scores, truth).item()

    assert new_value * 0 == 0
    assert old_value * 0 != 0
