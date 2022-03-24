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


class CrossEntDiceLoss(_Loss):
    """
    Weighted cross entropy + dice
    WARNING: the first dimension of inputs is batch size and is mandatory.
    Attributes
    ----------
    xent_weight : float, optional
        Cross entropy loss coefficient.
    dice_weight : float, optional
        Dice loss coefficient
    """

    def __init__(self, xent_weight=1.0, dice_weight=1.0, reduction="mean"):
        super(CrossEntDiceLoss, self).__init__(reduction=reduction)
        self.xent_weight = xent_weight
        self.dice_weight = dice_weight

    def forward(self, y_true, y_pred):
        return self.dice_weight * (
            1 - dice_coeff(y_true, y_pred)
        ) + self.xent_weight * balanced_xent(y_true, y_pred)


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

    # If both inputs are empty the dice coefficient should be equal 1
    dice[union == 0] = 1

    return torch.mean(dice)


def balanced_xent(y_true, y_pred, W=None):
    """
    Balanced xent.
    Parameters
    ----------
    y_pred : torch.Tensor
        predictions
    y_true : torch.Tensor
        targets
    W : float or None
        loss weight
    Returns
    -------
    dice : float
        Similarity coefficient in [0, 1].
    """
    eps = 1.0e-7
    y_pred = torch.clamp(y_pred.float(), min=eps, max=1.0 - eps)
    if W is None:
        w = 1 / torch.clamp(y_true.float().mean(), min=1.0e-7) - 1
    else:
        w = float(W)

    loss = -torch.mean(
        w * y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)
    )

    return loss
