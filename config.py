from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ModelConfig:
    # Model Architecture
    num_classes: int = 4
    pretrained_path: str = "medsam_vit_b.pth"
    freeze_encoder: bool = False
    
    # Image settings
    image_size: Tuple[int, int] = (128, 128)
    in_channels: int = 1
    
    # Attention settings
    attention_channels: int = 32
    se_reduction: int = 16
    
    # UNet settings
    unet_features: int = 64
    unet_levels: int = 4

@dataclass
class LossConfig:
    # Loss function weights
    lambda1: float = 0.5  # Focal Tversky Loss weight
    lambda2: float = 0.4  # DICE Loss weight  
    lambda3: float = 0.1  # Cross-Entropy Loss weight
    
    # Focal Tversky Loss parameters
    focal_alpha: float = 0.7  # weight for false positives
    focal_beta: float = 0.3   # weight for false negatives
    focal_gamma: float = 1.0  # focal parameter
    
    # DICE Loss parameters
    dice_smooth: float = 1e-6
    
    # Cross-Entropy Loss parameters
    use_class_weights: bool = True
    class_weights: Optional[Tuple[float, ...]] = (0.1, 0.3, 0.3, 0.3)  # background, myocardium, blood pool, scar
    
    # Adaptive loss settings
    adaptive_weights: bool = True

@dataclass
class TrainingConfig:
    # Basic training settings
    batch_size: int = 1
    num_workers: int = 6
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Optimizer settings
    beta1: float = 0.9
    beta2: float = 0.99
    
    # Learning rate scheduling
    lr_schedule_factor: float = 0.95
    lr_schedule_patience: int = 10
    
    # Gradient accumulation
    accumulation_steps: int = 4
    
    # Hardware settings
    gpu_id: str = "0"
    use_amp: bool = True  # Automatic Mixed Precision

@dataclass
class DataConfig:
    # Data paths
    data_path: str = "../Segmentation_LGE/Data"
    train_path: str = "Training"
    test_path: str = "Testing"
    x_folder: str = "Mag_image"
    y_folder: str = "4layer_mask"
    
    # Data processing
    normalize: bool = True
    augment: bool = True
    
    # Train-test split
    test_size: float = 0.25
    random_seed: int = 42

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    loss: LossConfig = LossConfig()
    
    # Logging and checkpoints
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_frequency: int = 5
    
    # Output visualization
    visualize_results: bool = True
    max_visualizations: int = 10

    def __post_init__(self):
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create config from dictionary."""
        return cls(
            ModelConfig(**config_dict.get('model', {})),
            TrainingConfig(**config_dict.get('training', {})),
            DataConfig(**config_dict.get('data', {})),
            LossConfig(**config_dict.get('loss', {}))
        )
    
    def from_yaml(self, yaml_path: str):
        """Load configuration from YAML file."""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update current instance with loaded values
        loaded_config = self.from_dict(config_dict)
        self.model = loaded_config.model
        self.training = loaded_config.training
        self.data = loaded_config.data
        self.loss = loaded_config.loss

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'loss': self.loss.__dict__
        }

# Default configuration instance
config = Config()