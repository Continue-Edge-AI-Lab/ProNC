import torch
import torch.nn as nn

class DRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat, target, avg_factor=None):
        assert avg_factor is None
        dot = torch.sum(feat * target, dim=1)
        loss = 0.5 * torch.mean((dot - 1) ** 2)
        return loss 