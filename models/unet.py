import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention.se_layer import SELayer
from .attention.scar_attention import ScarAttention

class UNetBlock(nn.Module):
    """Enhanced UNet block with attention mechanisms."""
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(out_channels)
        self.scar_attention = ScarAttention(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.se(x)
        x = self.scar_attention(x)
        return x

class UNet(nn.Module):
    """
    Modified UNet architecture with attention mechanisms and skip connections.
    Designed specifically for cardiac scar segmentation.
    """
    def __init__(self, num_classes=4, features=64):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = UNetBlock(1, features)
        self.enc2 = UNetBlock(features, features*2)
        self.enc3 = UNetBlock(features*2, features*4)
        self.enc4 = UNetBlock(features*4, features*8)
        
        # Center
        self.center = UNetBlock(features*8, features*16)
        
        # Decoder
        self.dec4 = UNetBlock(features*16 + features*8, features*8)
        self.dec3 = UNetBlock(features*8 + features*4, features*4)
        self.dec2 = UNetBlock(features*4 + features*2, features*2)
        self.dec1 = UNetBlock(features*2 + features, features)
        
        # Final convolution
        self.final = nn.Conv2d(features, num_classes, 1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Center
        center = self.center(self.pool(enc4))
        
        # Decoder path with skip connections
        dec4 = self.dec4(torch.cat([
            F.interpolate(center, enc4.shape[2:], mode='bilinear', align_corners=False),
            enc4
        ], 1))
        
        dec3 = self.dec3(torch.cat([
            F.interpolate(dec4, enc3.shape[2:], mode='bilinear', align_corners=False),
            enc3
        ], 1))
        
        dec2 = self.dec2(torch.cat([
            F.interpolate(dec3, enc2.shape[2:], mode='bilinear', align_corners=False),
            enc2
        ], 1))
        
        dec1 = self.dec1(torch.cat([
            F.interpolate(dec2, enc1.shape[2:], mode='bilinear', align_corners=False),
            enc1
        ], 1))
        
        return self.final(dec1)

    def initialize_weights(self):
        """Initialize model weights for better training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)