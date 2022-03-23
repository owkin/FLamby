import torch
from torch.nn.modules.loss import _Loss


class DiceLoss(_Loss):
    """
    Dice loss for segmentation.
    WARNING: the first dimension of inputs is batch size and is mandatory.
    """

    def __init__(self, reduction="mean"):
        super(DiceLoss, self).__init__(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return 1 - dice_coeff(input, target)


def dice_coeff(y_pred, y_true):
    """
    Return the dice coefficient.
    WARNING: the first dimension of inputs is batch size and is mandatory.
    Parameters
    ----------
    y_pred : torch.Tensor
        predictions
    y_true : torch.Tensor
        targets
    Returns
    -------
    dice : float
        Similarity coefficient in [0, 1].
    """
    intersection = torch.sum(y_pred * y_true, dim=tuple(range(1, y_true.dim())))

    union = torch.sum(0.5 * (y_pred + y_true), dim=tuple(range(1, y_true.dim())))

    dice = intersection / torch.clamp(union, min=1.0e-7)

    return torch.mean(dice)
