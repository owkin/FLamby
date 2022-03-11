import torch.nn as nn
from torch.nn import functional as F
import torch


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        #self.alpha = torch.tensor([alpha, 1-alpha]) #.cuda() RL: pass the device as argument ?
        self.alpha = torch.tensor([1,1,1,1,1,1,1,1,1]) # RL
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
