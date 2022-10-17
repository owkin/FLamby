import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class BaselineLoss(_Loss):
    """
    Weighted cross entropy + dice
    Input should have shape (B, C, ...): batch size x channels x any number of dimensions
    Attributes
    ----------
    xent_weight : float, optional
        Cross entropy loss coefficient.
    dice_weight : float, optional
        Dice loss coefficient
    """

    def __init__(
        self, xent_weight=0.1, dice_weight=1.0, alpha=0.5, beta=0.5, reduction="mean"
    ):
        super(BaselineLoss, self).__init__(reduction=reduction)
        self.xent_weight = xent_weight
        self.dice_weight = dice_weight
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        return self.dice_weight * (
            1 - dice_coeff(y_pred, y_true, self.alpha, self.beta)
        ) + self.xent_weight * balanced_xent(y_pred, y_true)


class DiceLoss(_Loss):
    """
    Dice loss for segmentation.
    If alpha and beta are not equal to 0.5, implements the Tversky loss.
    Input should have shape (B, C, ...): batch size x channels x any number of dimensions
    Attributes
    ----------
    alpha: float, default=.5
        first coefficient of Tversky loss
    beta: float, default=.5
        second coefficient of Tversky loss
    squared: bool, default=False
        Whether to square the summands in the denominator. Supersedes alpha and beta.
    """

    def __init__(self, alpha=0.5, beta=0.5, reduction="mean", squared=False):
        super(DiceLoss, self).__init__(reduction=reduction)
        self.alpha = alpha
        self.beta = beta
        self.squared = squared

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return 1 - dice_coeff(input, target, self.alpha, self.beta, self.squared)


class FocalLoss(_Loss):
    """
    Focal loss for dense detection: https://arxiv.org/abs/1708.02002.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__(reduction=reduction)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return _focal_loss(input, target, self.alpha, self.gamma, self.reduction)


def dice_coeff(y_pred, y_true, alpha=0.5, beta=0.5, squared=False):
    """
    Return the dice coefficient.
    Input should have shape (B, C, ...): batch size x channels x any number of dimensions
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
    intersection = torch.sum(
        y_pred * y_true.type(torch.FloatTensor).cuda(), dim=tuple(range(1, y_true.ndim))
    )

    if squared:
        union = 0.5 * torch.sum(
            y_pred * y_pred
            + y_true.type(torch.FloatTensor).cuda()
            * y_true.type(torch.FloatTensor).cuda(),
            dim=tuple(range(1, y_true.ndim)),
        )
    else:
        union = torch.sum(
            y_pred * y_true.type(torch.FloatTensor).cuda()
            + alpha * y_pred * (1 - y_true.type(torch.FloatTensor).cuda())
            + beta * (1 - y_pred) * y_true.type(torch.FloatTensor).cuda(),
            dim=tuple(range(1, y_true.ndim)),
        )

    dice = intersection / torch.clamp(union, min=1.0e-7)

    # If both inputs are empty the dice coefficient should be equal to 1
    dice[union == 0] = 1

    return torch.mean(dice)


def balanced_xent(y_pred, y_true, W=None):
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
        w * y_true.type(torch.FloatTensor).cuda() * torch.log(y_pred)
        + (1 - y_true.type(torch.FloatTensor).cuda()) * torch.log(1 - y_pred)
    )

    return loss


def _focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Arguments
    ---------
    inputs: A float tensor of arbitrary shape.
            The probability predictions for each example.
    targets: A float tensor with the same shape as inputs. Stores the binary
             classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
    alpha: (optional) Weighting factor in range (0,1) to balance
            positive vs negative examples. Default = -1 (no weighting).
    gamma: Exponent of the modulating factor (1 - p_t) to
           balance easy vs hard examples.
    reduction: 'none' | 'mean' | 'sum'
             'none': No reduction will be applied to the output.
             'mean': The output will be averaged.
             'sum': The output will be summed.
    Returns
    -------
        Loss tensor with the reduction option applied.
    """
    p = inputs.float()
    targets = targets.float()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
