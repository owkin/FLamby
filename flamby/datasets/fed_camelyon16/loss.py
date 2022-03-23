from torch.nn.modules.loss import _Loss
import torch


class BaselineLoss(_Loss):
    def __init__(self, reduction="mean"):
        super(BaselineLoss, self).__init__(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        y_pred = input
        bce = torch.nn.BCEWithLogitsLoss()
        return bce(y_pred, target)
