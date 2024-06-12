import torch
from torch.nn.modules.loss import _Loss


class BaselineLoss(_Loss):
    def __init__(self):
        super(BaselineLoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return self.ce(input, target.squeeze(axis=1).long())
