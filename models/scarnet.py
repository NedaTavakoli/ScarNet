import torch
import torch.nn as nn
import torch.nn.functional as F
from .medsam import MedSAMEncoder
from .unet import UNet
from .attention.scar_attention import ScarAttention, MultiScaleScarAttention
from .attention.se_layer import SELayer

class ScarNet(nn.Module):
    """
    ScarNet: A hybrid architecture combining MedSAM and UNet for cardiac scar segmentation.
    """
    def __init__(self, pretrained_path, num_classes=4, freeze_encoder=False):
        super(ScarNet, self).__init__()
        
        # Initialize both pathways
        self.medsam_encoder = MedSAMEncoder(
            pretrained_path=pretrained_path,
            freeze_encoder=freeze_encoder
        )
        self.unet = UNet(num_classes=num_classes)
        
        # Feature fusion components
        self.channel_reducer = nn.Conv2d(224, 32, kernel_size=1)  # 224 = 32 + 64 + 128
        self.scar_attention = MultiScaleScarAttention(32)
        
        # Final layers
        self.fusion = nn.Sequential(
            nn.Conv2d(num_classes * 2, num_classes * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_classes * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes * 2, num_classes, kernel_size=1)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize the weights of non-pretrained parts."""
        for m in [self.channel_reducer, self.fusion]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through ScarNet.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, 1, H, W]
        
        Returns:
            torch.Tensor: Segmentation output of shape [B, num_classes, H, W]
        """
        # MedSAM pathway
        medsam_features = self.medsam_encoder(x)
        
        # Combine multi-scale MedSAM features
        combined_features = torch.cat(medsam_features, dim=1)
        reduced_features = self.channel_reducer(combined_features)
        
        # Apply scar-specific attention
        attended_features = self.scar_attention(reduced_features)
        
        # UNet pathway
        unet_output = self.unet(x)
        
        # Ensure both outputs have the same size
        if attended_features.size()[2:] != unet_output.size()[2:]:
            attended_features = F.interpolate(
                attended_features,
                size=unet_output.size()[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Combine outputs
        combined = torch.cat([
            F.softmax(self.medsam_to_classes(attended_features), dim=1),
            F.softmax(unet_output, dim=1)
        ], dim=1)
        
        # Final fusion and segmentation
        output = self.fusion(combined)
        
        return output

    def medsam_to_classes(self, features):
        """Convert MedSAM features to class predictions."""
        return nn.Conv2d(features.size(1), self.unet.final.out_channels, 1).to(features.device)(features)

    def save(self, path):
        """Save model weights."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_classes': self.unet.final.out_channels
        }, path)

    @classmethod
    def load(cls, path, map_location=None):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(
            pretrained_path=None,  # Don't load pretrained weights
            num_classes=checkpoint['num_classes']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

class ScarNetWithFeatures(ScarNet):
    """
    Extended version of ScarNet that also returns intermediate features
    for visualization or additional analysis.
    """
    def forward(self, x):
        # MedSAM pathway with features
        medsam_features = self.medsam_encoder(x)
        combined_features = torch.cat(medsam_features, dim=1)
        reduced_features = self.channel_reducer(combined_features)
        attended_features = self.scar_attention(reduced_features)
        
        # UNet pathway
        unet_output = self.unet(x)
        
        # Size matching
        if attended_features.size()[2:] != unet_output.size()[2:]:
            attended_features = F.interpolate(
                attended_features,
                size=unet_output.size()[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Combine outputs
        combined = torch.cat([
            F.softmax(self.medsam_to_classes(attended_features), dim=1),
            F.softmax(unet_output, dim=1)
        ], dim=1)
        
        # Final fusion
        output = self.fusion(combined)
        
        # Return output and intermediate features
        return {
            'output': output,
            'medsam_features': medsam_features,
            'attended_features': attended_features,
            'unet_output': unet_output,
            'combined_features': combined
        }