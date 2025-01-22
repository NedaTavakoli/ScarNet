import torch
import numpy as np
import random
import torchvision.transforms.functional as TF
from torchvision import transforms
import cv2

class Augmentation:
    """
    Data augmentation pipeline for cardiac MRI images and masks.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        """
        Apply augmentations to image and mask pair.
        
        Args:
            image (torch.Tensor): Input image [1, H, W]
            mask (torch.Tensor): Segmentation mask [H, W]
        """
        # Convert to PIL for torchvision transforms
        if random.random() < self.p:
            # Random rotation
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

            # Random horizontal flip
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random vertical flip
            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Random brightness and contrast
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
            
            # Random Gaussian noise
            if random.random() < 0.5:
                noise = torch.randn_like(image) * 0.05
                image = image + noise
                image = torch.clamp(image, 0, 1)

        return image, mask

class AdvancedAugmentation:
    """
    Advanced augmentation techniques specific to cardiac MRI.
    """
    def __init__(self, p=0.5):
        self.p = p
        
    def elastic_transform(self, image, mask, alpha=100, sigma=10):
        """Apply elastic deformation."""
        if random.random() > self.p:
            return image, mask
            
        shape = image.shape[1:]
        dx = np.random.rand(*shape) * 2 - 1
        dy = np.random.rand(*shape) * 2 - 1
        
        dx = cv2.GaussianBlur(dx, (0,0), sigma)
        dy = cv2.GaussianBlur(dy, (0,0), sigma)
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy*alpha, (-1, 1)), np.reshape(x+dx*alpha, (-1, 1))
        
        def apply_transform(img):
            return torch.from_numpy(
                map_coordinates(img.numpy(), indices, order=1).reshape(shape)
            )
        
        return apply_transform(image), apply_transform(mask)
    
    def gamma_correction(self, image, gamma_range=(0.8, 1.2)):
        """Apply random gamma correction."""
        if random.random() > self.p:
            return image
            
        gamma = random.uniform(*gamma_range)
        return image.pow(gamma)
    
    def __call__(self, image, mask):
        """Apply all advanced augmentations."""
        # Apply standard augmentations
        image, mask = Augmentation(self.p)(image, mask)
        
        # Apply elastic transform
        image, mask = self.elastic_transform(image, mask)
        
        # Apply gamma correction to image only
        image = self.gamma_correction(image)
        
        return image, mask

class AugmentationPipeline:
    """Complete augmentation pipeline combining all transformations."""
    def __init__(self, use_advanced=False):
        self.basic_aug = Augmentation(p=0.5)
        self.advanced_aug = AdvancedAugmentation(p=0.3) if use_advanced else None
        
        # Normalization parameters
        self.normalize = transforms.Normalize(
            mean=[0.485],
            std=[0.229]
        )
        
    def __call__(self, image, mask):
        # Basic augmentations
        image, mask = self.basic_aug(image, mask)
        
        # Advanced augmentations if enabled
        if self.advanced_aug is not None:
            image, mask = self.advanced_aug(image, mask)
        
        # Normalize image
        image = self.normalize(image)
        
        return image, mask

def get_augmentation(config):
    """Factory function to create augmentation based on config."""
    return AugmentationPipeline(
        use_advanced=config.data.use_advanced_augmentation
    )