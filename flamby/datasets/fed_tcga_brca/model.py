import torch
import torch.nn as nn

from flamby.datasets.fed_tcga_brca import FedTcgaBrca


class Baseline(nn.Module):
    """
    Baseline model: a linear layer !
    """

    def __init__(self):
        super(Baseline, self).__init__()
        input_size = 39
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        out = self.fc(x)
        return out


if __name__ == "__main__":

    mydataset = FedTcgaBrca(train=True, pooled=True)

    model = Baseline()

    for i in range(10):
        X = torch.unsqueeze(mydataset[i][0], 0)
        y = torch.unsqueeze(mydataset[i][1], 0)
        print(X.shape)
        print(y)
        print(model(X))
