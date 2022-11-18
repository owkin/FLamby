import torch
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, Dataset
from torchvision import models


class FedDummyDataset(Dataset):
    def __init__(
        self,
        center=0,
        train=True,
        pooled=False,
        X_dtype=torch.float32,
        y_dtype=torch.float32,
        debug=False,
    ):
        super().__init__()
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.size = (center + 1) * 10 * 42
        self.centers = center

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (
            torch.rand(3, 224, 224).to(self.X_dtype),
            torch.randint(0, 2, (1,)).to(self.y_dtype),
        )


class BaselineLoss(_Loss):
    def __init__(self, reduction="mean"):
        super(BaselineLoss, self).__init__(reduction=reduction)
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return self.bce(input, target)


class Baseline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.architecture = models.mobilenet_v2(pretrained=False)
        nftrs = [m for m in self.architecture.classifier.modules()][-1].in_features
        self.architecture.classifier = torch.nn.Linear(nftrs, 1)

    def forward(self, X):
        return self.architecture(X)


if __name__ == "__main__":
    m = Baseline()
    lo = BaselineLoss()
    dl = DataLoader(
        FedDummyDataset(center=1, train=True), batch_size=32, shuffle=True, num_workers=0
    )
    it = iter(dl)
    X, y = next(it)
    opt = torch.optim.SGD(m.parameters(), lr=1.0)
    y_pred = m(X)
    ls = lo(y_pred, y)
    ls.backward()
    opt.step()
