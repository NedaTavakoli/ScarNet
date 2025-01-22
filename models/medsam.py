import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from .attention.scar_attention import ScarAttention
from .attention.se_layer import SELayer

class MedSAMEncoder(nn.Module):
    """MedSAM-based encoder with enhanced attention mechanisms."""
    
    def __init__(self, pretrained_path, freeze_encoder=False):
        super(MedSAMEncoder, self).__init__()
        # Initialize MedSAM
        self.model_type = "vit_b"
        sam = sam_model_registry[self.model_type]()
        self.image_encoder = sam.image_encoder
        
        # Load pretrained weights
        if pretrained_path:
            state_dict = torch.load(pretrained_path)
            self.image_encoder.load_state_dict(state_dict, strict=True)
        
        if freeze_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        
        # Additional processing layers
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                SELayer(128),
                ScarAttention(128)
            ),
            nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                SELayer(64),
                ScarAttention(64)
            ),
            nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                SELayer(32),
                ScarAttention(32)
            )
        ])
        
        self.upsampler = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)

    def interpolate_pos_encoding(self, x, pos_embed):
        """Interpolate positional encoding to match input size."""
        N, H, W, C = x.shape
        if len(pos_embed.shape) == 3:
            pos_embed = pos_embed.reshape(1, int(pow(pos_embed.shape[1], 0.5)), 
                                       int(pow(pos_embed.shape[1], 0.5)), C)
        pos_embed = nn.functional.interpolate(
            pos_embed.permute(0, 3, 1, 2),
            size=(H, W),
            mode='bicubic',
            align_corners=False
        ).permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x):
        # Handle grayscale input
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Patch embedding
        x = self.image_encoder.patch_embed(x)
        
        # Add positional encoding
        pos_embed = self.interpolate_pos_encoding(x, self.image_encoder.pos_embed)
        x = x + pos_embed
        
        # Process through transformer blocks
        for blk in self.image_encoder.blocks:
            x = blk(x)
        
        # Process through neck
        x = self.image_encoder.neck(x.permute(0, 3, 1, 2))
        
        # Process through decoder blocks
        features = []
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)
            features.append(self.upsampler(x))
        
        return features