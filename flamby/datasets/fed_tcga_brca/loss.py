import torch
from dataset import FedTcgaBrca
from model import Baseline


def BaselineLoss(scores, truth):
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

    # The Cox loss expects events to be reverse sorted in time
    a = torch.stack((torch.squeeze(scores), truth[:, 0], truth[:, 1]), dim=1)
    a = torch.stack(sorted(a, key=lambda a: -a[2]))

    scores = a[:, 0]
    events = a[:, 1]

    scores_ = scores - scores.max()
    loss = -(scores_ - torch.log(torch.exp(scores_).cumsum(0))) * events
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

    print(BaselineLoss(scores, truth))
