import torch
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self, input_dim=10, output_dim=3):
        super(Baseline, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))
