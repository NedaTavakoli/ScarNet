import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from pathlib import Path
from .augmentation import get_augmentation
import logging

class CardiacDataset(Dataset):
    """
    Dataset class for loading cardiac MRI data.
    """
    def __init__(self, x_files, y_files, imsize=128, augment=False, config=None):
        self.x_files = x_files
        self.y_files = y_files
        self.imsize = imsize
        self.augment = augment
        self.augmentation = get_augmentation(config) if augment else None
        
        # Verify files exist
        self._verify_files()
        
    def _verify_files(self):
        """Verify all files exist."""
        for x_file, y_file in zip(self.x_files, self.y_files):
            if not Path(x_file).exists():
                raise FileNotFoundError(f"Image file not found: {x_file}")
            if not Path(y_file).exists():
                raise FileNotFoundError(f"Mask file not found: {y_file}")

    def __len__(self):
        return len(self.x_files)

    def normalize(self, data):
        """Normalize the data to [0,1] range."""
        return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

    def __getitem__(self, idx):
        try:
            # Load image
            with h5py.File(self.x_files[idx], 'r') as f:
                x = np.array(f['image_mag'])
            
            # Load mask
            with h5py.File(self.y_files[idx], 'r') as f:
                y = np.array(f['mask'])
            
            # Normalize and convert to tensor
            x = self.normalize(x)
            x = torch.from_numpy(x).float().unsqueeze(0)
            y = torch.from_numpy(y).long()
            
            # Add channel dimension to mask if needed
            if y.dim() == 2:
                y = y.unsqueeze(0)
            
            # Resize to target size
            x = TF.resize(x, (self.imsize, self.imsize), antialias=True)
            y = TF.resize(y.float(), (self.imsize, self.imsize), antialias=False).long()
            
            # Apply augmentation if enabled
            if self.augment and self.augmentation is not None:
                x, y = self.augmentation(x, y)
            
            return {'image': x, 'mask': y, 'path': self.x_files[idx]}
            
        except Exception as e:
            logging.error(f"Error loading file {self.x_files[idx]}: {str(e)}")
            raise

class CardiacDataModule:
    """
    Data module for managing cardiac datasets and dataloaders.
    """
    def __init__(self, config):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage=None):
        """Setup train, validation and test datasets."""
        if stage == 'fit' or stage is None:
            # Get file lists
            x_files = sorted(Path(self.config.data.train_path).glob('**/Mag_image/*.h5'))
            y_files = [Path(str(x).replace('Mag_image', '4layer_mask')) for x in x_files]
            
            # Split into train/val
            train_size = int(0.8 * len(x_files))
            train_x, val_x = x_files[:train_size], x_files[train_size:]
            train_y, val_y = y_files[:train_size], y_files[train_size:]
            
            # Create datasets
            self.train_dataset = CardiacDataset(
                train_x, train_y,
                imsize=self.config.data.image_size,
                augment=True,
                config=self.config
            )
            
            self.val_dataset = CardiacDataset(
                val_x, val_y,
                imsize=self.config.data.image_size,
                augment=False,
                config=self.config
            )
            
        if stage == 'test' or stage is None:
            # Setup test dataset
            test_x = sorted(Path(self.config.data.test_path).glob('**/Mag_image/*.h5'))
            test_y = [Path(str(x).replace('Mag_image', '4layer_mask')) for x in test_x]
            
            self.test_dataset = CardiacDataset(
                test_x, test_y,
                imsize=self.config.data.image_size,
                augment=False,
                config=self.config
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=True
        )