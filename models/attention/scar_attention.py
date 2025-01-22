import torch
import torch.nn as nn
import torch.nn.functional as F

class ScarAttention(nn.Module):
    """
    Specialized attention mechanism for scar regions in cardiac MRI.
    Uses spatial and channel attention to focus on scar-relevant features.
    """
    def __init__(self, in_channels):
        super(ScarAttention, self).__init__()
        self.channel_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Apply channel and spatial attention to input features."""
        # Channel attention
        channel_att = self.channel_gate(F.adaptive_avg_pool2d(x, 1))
        x = x * channel_att
        
        # Spatial attention
        spatial_att = self.spatial_gate(x)
        x = x * spatial_att
        
        return x

class MultiScaleScarAttention(nn.Module):
    """Multi-scale attention for capturing scar features at different scales."""
    def __init__(self, in_channels):
        super(MultiScaleScarAttention, self).__init__()
        self.scales = [1, 2, 4]
        self.attentions = nn.ModuleList([
            ScarAttention(in_channels) for _ in self.scales
        ])
        self.fusion = nn.Conv2d(in_channels * len(self.scales), in_channels, 1)

    def forward(self, x):
        """Process features at multiple scales and fuse results."""
        outputs = []
        for scale, attention in zip(self.scales, self.attentions):
            # Scale input if needed
            if scale > 1:
                scaled = F.adaptive_avg_pool2d(x, (x.shape[2] // scale, x.shape[3] // scale))
            else:
                scaled = x
            
            # Apply attention
            attended = attention(scaled)
            
            # Restore original size if needed
            if scale > 1:
                attended = F.interpolate(attended, size=(x.shape[2], x.shape[3]), 
                                      mode='bilinear', align_corners=False)
            outputs.append(attended)
        
        # Concatenate and fuse
        return self.fusion(torch.cat(outputs, dim=1))