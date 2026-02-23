import torch
import torch.nn as nn
import torch.nn.functional as F


class L2NormLoss(nn.Module):

    def __init__(self, reduction='mean'):

        super(L2NormLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, source, target):

        # Compute squared L2 norm (MSE)
        loss = F.mse_loss(source, target, reduction=self.reduction)
        
        return loss
    
    
# Alternative implementation with more control
class L2NormLossDetailed(nn.Module):

    def __init__(self, normalize_features=False):

        super(L2NormLossDetailed, self).__init__()
        self.normalize_features = normalize_features
        
    def forward(self, source, target):
        
        # Compute squared element-wise differences
        squared_diff = 0.5*(source - target) ** 2
        
        # Sum across feature dimensions
        squared_distances = squared_diff.sum(dim=1)
        
        # Apply reduction

        return squared_distances.sum()
