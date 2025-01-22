import torch
import torch.nn as nn

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Layer for channel attention.
    Paper: https://arxiv.org/abs/1709.01507
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)

class AdaptiveSELayer(nn.Module):
    """
    Adaptive Squeeze-and-Excitation Layer with dynamic reduction ratio.
    """
    def __init__(self, channel, reduction_bounds=(8, 32)):
        super(AdaptiveSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        min_reduction, max_reduction = reduction_bounds
        
        # Calculate adaptive reduction based on channel count
        reduction = min(max(channel // 16, min_reduction), max_reduction)
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)