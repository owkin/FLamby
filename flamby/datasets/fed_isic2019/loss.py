# Our baseline loss is the weighted focal loss.
# Focal loss was first implemented by He et al. in the following [article]
# (https://arxiv.org/abs/1708.02002)
# Thank you to [Aman Arora](https://github.com/amaarora) for this nice [explanation]
# (https://amaarora.github.io/2020/06/29/FocalLoss.html)


import random

import albumentations
import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from flamby.datasets.fed_isic2019 import Baseline, FedIsic2019


class BaselineLoss(_Loss):
    """Weighted focal loss
    See this [link](https://amaarora.github.io/2020/06/29/FocalLoss.html) for
    a good explanation
    Attributes
    ----------
    alpha: torch.tensor of size 8, class weights
    gamma: torch.tensor of size 1, positive float, for gamma = 0 focal loss is
    the same as CE loss, increases gamma reduces the loss for the "hard to classify
    examples"
    """

    def __init__(
        self,
        alpha=torch.tensor(
            [5.5813, 2.0472, 7.0204, 26.1194, 9.5369, 101.0707, 92.5224, 38.3443]
        ),
        gamma=2.0,
    ):
        super(BaselineLoss, self).__init__()
        self.alpha = alpha.to(torch.float)
        self.gamma = gamma

    def forward(self, inputs, targets):
        """Weighted focal loss function
        Parameters
        ----------
        inputs : torch.tensor of size 8, logits output by the model (pre-softmax)
        targets : torch.tensor of size 1, int between 0 and 7, groundtruth class
        """
        targets = targets.view(-1, 1).type_as(inputs)
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.long())
        logpt = logpt.view(-1)
        pt = logpt.exp()
        self.alpha = self.alpha.to(targets.device)
        at = self.alpha.gather(0, targets.data.view(-1).long())
        logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt

        return loss.mean()


if __name__ == "__main__":

    sz = 200
    train_aug = albumentations.Compose(
        [
            albumentations.RandomScale(0.07),
            albumentations.Rotate(50),
            albumentations.RandomBrightnessContrast(0.15, 0.1),
            albumentations.Flip(p=0.5),
            albumentations.Affine(shear=0.1),
            albumentations.RandomCrop(sz, sz),
            albumentations.CoarseDropout(random.randint(1, 8), 16, 16),
            albumentations.Normalize(always_apply=True),
        ]
    )

    mydataset = FedIsic2019(0, train=True, pooled=True, augmentations=train_aug)

    loss = BaselineLoss(alpha=torch.tensor([1, 2, 1, 1, 5, 1, 1, 1]))
    model = Baseline()

    for i in range(10):
        X = torch.unsqueeze(mydataset[i][0], 0)
        y = torch.unsqueeze(mydataset[i][1], 0)

        y_hat = model(X)
        print(X.shape, type(X))
        print(y_hat.shape, type(y_hat))
        print(y.shape, type(y))

        print(loss(y_hat, y))
