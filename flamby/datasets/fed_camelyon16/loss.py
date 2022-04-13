import torch
from torch.nn.modules.loss import _Loss


class BaselineLoss(_Loss):
    def __init__(self, reduction="mean"):
        super(BaselineLoss, self).__init__(reduction=reduction)
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return self.bce(input, target)
