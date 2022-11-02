import torch
import torch.nn as nn

from flamby.datasets.fed_tcga_brca import Baseline, FedTcgaBrca


class BaselineLoss(nn.Module):
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
        super(BaselineLoss, self).__init__()

    def forward(self, scores, truth):
        # The Cox loss calc expects events to be reverse sorted in time
        a = torch.stack((torch.squeeze(scores, dim=1), truth[:, 0], truth[:, 1]), dim=1)
        a = torch.stack(sorted(a, key=lambda a: -a[2]))
        scores = a[:, 0]
        events = a[:, 1]
        loss = torch.zeros(scores.size(0)).to(device=scores.device, dtype=scores.dtype)
        for i in range(1, scores.size(0)):
            aux = scores[: i + 1] - scores[i]
            m = aux.max()
            aux_ = aux - m
            aux_.exp_()
            loss[i] = m + torch.log(aux_.sum(0))
        loss *= events
        return loss.mean()


if __name__ == "__main__":

    mydataset = FedTcgaBrca(train=True, pooled=True)

    model = Baseline()

    X = torch.stack((mydataset[0][0], mydataset[1][0], mydataset[2][0]), dim=0)
    truth = torch.stack((mydataset[0][1], mydataset[1][1], mydataset[2][1]), dim=0)

    scores = model(X)

    print(X)
    print(scores)
    print(truth)

    loss = BaselineLoss()
    print(loss(scores, truth))
