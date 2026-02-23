import torch
import torch.nn as nn

class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) Loss with RBF kernel.
    Estimates squared MMD between two feature distributions source and target.
    """
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(MMDLoss, self).__init__()  # Correct super() call
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma  # This attribute is now properly initialized
        
    def gaussian_kernel(self, source, target):
        """
        Compute a sum of RBF kernels between source and target features.
        """
        # Concatenate along batch dimension
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        # Compute squared L2 distance matrix
        L2_distance = ((total0 - total1) ** 2).sum(2)

        # Determine bandwidth
        if self.fix_sigma is not None:
            bandwidth = self.fix_sigma
        else:
            n_samples = total.size(0)
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
            
        # Create multiple bandwidths - Fixed typo (kernel_mul**i)
        bandwidth_list = [bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)]
        
        # Sum RBF kernels with different bandwidths
        kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
        return sum(kernel_val)
        
    def forward(self, source, target):
        """
        Compute the MMD loss between source and target.
        """
        batch_size = source.size(0)
        kernels = self.gaussian_kernel(source, target)

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]

        loss = torch.mean(XX + YY - XY - YX)
        return loss