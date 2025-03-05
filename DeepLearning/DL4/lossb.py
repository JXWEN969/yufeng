import torch.nn as nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', gamma=0.25, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, y_pred, y_true):
        logp = self.ce(y_pred, torch.squeeze(y_true).long())
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return torch.Tensor(loss.mean())


