import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model output logits [B, C, H, W]
            targets: Ground truth labels [B, H, W]
        """
        # Convert predictions to probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot encoding
        num_classes = predictions.size(1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Flatten tensors
        predictions = predictions.view(predictions.size(0), predictions.size(1), -1)
        targets_one_hot = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)
        
        # Calculate Dice coefficient for each class
        intersection = (predictions * targets_one_hot).sum(dim=2)
        union = predictions.sum(dim=2) + targets_one_hot.sum(dim=2)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Return average Dice loss across all classes
        return 1 - dice.mean()


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss for handling class imbalance in segmentation."""
    
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.0, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha  # weight for false positives
        self.beta = beta    # weight for false negatives
        self.gamma = gamma  # focal parameter
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model output logits [B, C, H, W]
            targets: Ground truth labels [B, H, W]
        """
        # Convert predictions to probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot encoding
        num_classes = predictions.size(1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Flatten tensors
        predictions = predictions.view(predictions.size(0), predictions.size(1), -1)
        targets_one_hot = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)
        
        # Calculate Tversky index for each class
        true_pos = (predictions * targets_one_hot).sum(dim=2)
        false_neg = (targets_one_hot * (1 - predictions)).sum(dim=2)
        false_pos = ((1 - targets_one_hot) * predictions).sum(dim=2)
        
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )
        
        # Apply focal mechanism
        focal_tversky = (1 - tversky) ** self.gamma
        
        # Return average focal Tversky loss across all classes
        return focal_tversky.mean()


class CombinedLoss(nn.Module):
    """Combined loss function incorporating Focal Tversky, Dice, and Cross-Entropy losses."""
    
    def __init__(self, lambda1=0.5, lambda2=0.4, lambda3=0.1, 
                 focal_alpha=0.7, focal_beta=0.3, focal_gamma=1.0,
                 dice_smooth=1e-6, ce_weight=None):
        """
        Args:
            lambda1: Weight for Focal Tversky Loss
            lambda2: Weight for Dice Loss
            lambda3: Weight for Cross-Entropy Loss
            focal_alpha: Alpha parameter for Focal Tversky Loss
            focal_beta: Beta parameter for Focal Tversky Loss
            focal_gamma: Gamma parameter for Focal Tversky Loss
            dice_smooth: Smoothing parameter for Dice Loss
            ce_weight: Class weights for Cross-Entropy Loss
        """
        super(CombinedLoss, self).__init__()
        
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        
        self.focal_tversky_loss = FocalTverskyLoss(
            alpha=focal_alpha, 
            beta=focal_beta, 
            gamma=focal_gamma
        )
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_weight)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model output logits [B, C, H, W]
            targets: Ground truth labels [B, H, W]
        """
        # Calculate individual losses
        focal_tversky = self.focal_tversky_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        cross_entropy = self.ce_loss(predictions, targets)
        
        # Combine losses with specified weights
        combined_loss = (
            self.lambda1 * focal_tversky +
            self.lambda2 * dice +
            self.lambda3 * cross_entropy
        )
        
        return combined_loss, {
            'focal_tversky': focal_tversky.item(),
            'dice': dice.item(),
            'cross_entropy': cross_entropy.item(),
            'combined': combined_loss.item()
        }


class WeightedCombinedLoss(nn.Module):
    """
    Weighted Combined Loss that adjusts weights based on class frequencies.
    Useful for handling severe class imbalance in cardiac segmentation.
    """
    
    def __init__(self, lambda1=0.5, lambda2=0.4, lambda3=0.1,
                 class_weights=None, adaptive_weights=True):
        super(WeightedCombinedLoss, self).__init__()
        
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.adaptive_weights = adaptive_weights
        
        # Default class weights for cardiac segmentation (background, myocardium, blood pool, scar)
        if class_weights is None:
            class_weights = torch.tensor([0.1, 0.3, 0.3, 0.3])  # Lower weight for background
        
        self.focal_tversky_loss = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.0)
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model output logits [B, C, H, W]
            targets: Ground truth labels [B, H, W]
        """
        # Calculate individual losses
        focal_tversky = self.focal_tversky_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        cross_entropy = self.ce_loss(predictions, targets)
        
        # Adaptive weight adjustment based on scar presence
        if self.adaptive_weights:
            # Increase Focal Tversky weight when scar class (class 3) is present
            scar_present = (targets == 3).any()
            current_lambda1 = self.lambda1 * 1.5 if scar_present else self.lambda1
        else:
            current_lambda1 = self.lambda1
        
        # Combine losses with potentially adaptive weights
        combined_loss = (
            current_lambda1 * focal_tversky +
            self.lambda2 * dice +
            self.lambda3 * cross_entropy
        )
        
        return combined_loss, {
            'focal_tversky': focal_tversky.item(),
            'dice': dice.item(),
            'cross_entropy': cross_entropy.item(),
            'combined': combined_loss.item(),
            'lambda1_used': current_lambda1
        }
