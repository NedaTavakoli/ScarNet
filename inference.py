#!/usr/bin/env python3
"""
ScarNet Inference Script
========================

Automated myocardial scar segmentation in Late Gadolinium Enhancement (LGE) Cardiac MRI
using ScarNet - a hybrid architecture combining MedSAM's Vision Transformer encoder 
with a U-Net decoder.

Author: Neda Tavakoli
Email: neda.tavakoli@northwestern.edu
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from tqdm import tqdm
import nibabel as nib
from typing import Optional, Tuple, List, Union
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ScarNetDataset(Dataset):
    """Dataset class for ScarNet inference."""
    
    def __init__(self, image_dir: str, transform=None):
        """
        Initialize dataset for inference.
        
        Args:
            image_dir (str): Directory containing input images
            transform: Image transformations to apply
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Supported image extensions
        self.extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.nii', '.nii.gz']
        
        # Get all image files
        self.image_paths = []
        for ext in self.extensions:
            self.image_paths.extend(list(self.image_dir.glob(f'*{ext}')))
            self.image_paths.extend(list(self.image_dir.glob(f'*{ext.upper()}')))
        
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")
        
        logger.info(f"Found {len(self.image_paths)} images for inference")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get image for inference."""
        img_path = self.image_paths[idx]
        
        # Load image based on extension
        if img_path.suffix.lower() in ['.nii', '.gz']:
            # Load NIfTI files (common in medical imaging)
            img_data = nib.load(str(img_path)).get_fdata()
            if len(img_data.shape) == 3:
                img_data = img_data[:, :, img_data.shape[2] // 2]  # Take middle slice
            image = Image.fromarray((img_data * 255).astype(np.uint8))
        else:
            # Load standard image formats
            image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default preprocessing
            image = np.array(image)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Resize to standard size (adjust based on your model requirements)
            image = cv2.resize(image, (256, 256))
            
            # Normalize
            image = image.astype(np.float32) / 255.0
            
            # Convert to tensor
            image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        
        return {
            'image': image,
            'path': str(img_path),
            'filename': img_path.name
        }


class ScarNetModel(nn.Module):
    """
    ScarNet model architecture combining MedSAM's ViT encoder with U-Net decoder.
    This is a placeholder implementation - replace with your actual model architecture.
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        super(ScarNetModel, self).__init__()
        
        # Placeholder architecture - replace with your actual ScarNet implementation
        # This should include:
        # 1. MedSAM's Vision Transformer encoder
        # 2. U-Net decoder
        # 3. SE attention layers
        # 4. ScarAttention blocks
        
        # Example encoder (replace with MedSAM ViT)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Example decoder (replace with U-Net decoder)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, num_classes, 1),
        )
    
    def forward(self, x):
        # Encode
        features = self.encoder(x)
        
        # Decode
        output = self.decoder(features)
        
        return output


class ScarNetInference:
    """Main inference class for ScarNet."""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize ScarNet inference.
        
        Args:
            model_path (str): Path to trained model weights
            config_path (str, optional): Path to configuration file
            device (str): Device to use for inference ('cpu', 'cuda', 'auto')
        """
        self.model_path = model_path
        self.config_path = config_path
        self.device = self._get_device(device)
        
        # Load configuration
        self.config = self._load_config()
        
        # Load model
        self.model = self._load_model()
        
        logger.info(f"ScarNet inference initialized on {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        """Determine the appropriate device for inference."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        return torch.device(device)
    
    def _load_config(self) -> dict:
        """Load configuration from file."""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Default configuration
            config = {
                'model': {
                    'architecture': 'ScarNet',
                    'num_classes': 2,
                    'input_size': 256
                },
                'inference': {
                    'batch_size': 1,
                    'threshold': 0.5,
                    'save_probability_maps': True,
                    'save_visualizations': True
                }
            }
        
        return config
    
    def _load_model(self) -> nn.Module:
        """Load the trained ScarNet model."""
        # Initialize model
        model = ScarNetModel(
            num_classes=self.config['model'].get('num_classes', 2),
            pretrained=False
        )
        
        # Load weights
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            
            logger.info(f"Loaded model weights from {self.model_path}")
        else:
            logger.warning(f"Model file {self.model_path} not found. Using random weights.")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for inference."""
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize
        target_size = self.config['model'].get('input_size', 256)
        image = cv2.resize(image, (target_size, target_size))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        
        return image.to(self.device)
    
    def postprocess_output(self, output: torch.Tensor, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess model output."""
        # Apply softmax if multi-class
        if output.shape[1] > 1:
            probs = torch.softmax(output, dim=1)
            scar_prob = probs[0, 1].cpu().numpy()  # Assuming class 1 is scar
        else:
            probs = torch.sigmoid(output)
            scar_prob = probs[0, 0].cpu().numpy()
        
        # Create binary mask
        binary_mask = (scar_prob > threshold).astype(np.uint8)
        
        return scar_prob, binary_mask
    
    def calculate_metrics(self, prob_map: np.ndarray, binary_mask: np.ndarray) -> dict:
        """Calculate scar quantification metrics."""
        total_pixels = prob_map.shape[0] * prob_map.shape[1]
        scar_pixels = np.sum(binary_mask)
        
        metrics = {
            'scar_area_pixels': int(scar_pixels),
            'total_area_pixels': int(total_pixels),
            'scar_percentage': float(scar_pixels / total_pixels * 100),
            'mean_scar_probability': float(np.mean(prob_map[binary_mask == 1])) if scar_pixels > 0 else 0.0,
            'max_scar_probability': float(np.max(prob_map)),
            'scar_volume_ml': float(scar_pixels * 0.1)  # Assuming pixel spacing (adjust as needed)
        }
        
        return metrics
    
    def visualize_results(self, image: np.ndarray, prob_map: np.ndarray, 
                         binary_mask: np.ndarray, filename: str) -> np.ndarray:
        """Create visualization of segmentation results."""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Probability map
        im1 = axes[1].imshow(prob_map, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title('Scar Probability Map')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Binary mask
        axes[2].imshow(binary_mask, cmap='gray')
        axes[2].set_title('Binary Scar Mask')
        axes[2].axis('off')
        
        # Overlay
        overlay = image.copy()
        overlay = np.stack([overlay] * 3, axis=-1)  # Convert to RGB
        overlay[binary_mask == 1] = [1.0, 0.0, 0.0]  # Red for scar
        axes[3].imshow(overlay)
        axes[3].set_title('Scar Overlay')
        axes[3].axis('off')
        
        plt.suptitle(f'ScarNet Results: {filename}')
        plt.tight_layout()
        
        # Convert plot to numpy array
        fig.canvas.draw()
        vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return vis_array
    
    def predict_single(self, image_path: str, output_dir: str) -> dict:
        """Run inference on a single image."""
        # Load and preprocess image
        if image_path.lower().endswith(('.nii', '.nii.gz')):
            img_data = nib.load(image_path).get_fdata()
            if len(img_data.shape) == 3:
                img_data = img_data[:, :, img_data.shape[2] // 2]
            image = (img_data * 255).astype(np.uint8)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        original_size = image.shape
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Postprocess
        threshold = self.config['inference'].get('threshold', 0.5)
        prob_map, binary_mask = self.postprocess_output(output, threshold)
        
        # Resize back to original size
        prob_map = cv2.resize(prob_map, (original_size[1], original_size[0]))
        binary_mask = cv2.resize(binary_mask, (original_size[1], original_size[0]))
        
        # Calculate metrics
        metrics = self.calculate_metrics(prob_map, binary_mask)
        
        # Save results
        filename = Path(image_path).stem
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save probability map
        if self.config['inference'].get('save_probability_maps', True):
            prob_path = output_path / f"{filename}_probability.png"
            cv2.imwrite(str(prob_path), (prob_map * 255).astype(np.uint8))
        
        # Save binary mask
        mask_path = output_path / f"{filename}_mask.png"
        cv2.imwrite(str(mask_path), binary_mask * 255)
        
        # Save visualization
        if self.config['inference'].get('save_visualizations', True):
            vis_array = self.visualize_results(
                image / 255.0, prob_map, binary_mask, filename
            )
            vis_path = output_path / f"{filename}_visualization.png"
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis_array, cv2.COLOR_RGB2BGR))
        
        # Save metrics
        metrics_path = output_path / f"{filename}_metrics.yaml"
        with open(metrics_path, 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)
        
        logger.info(f"Processed {filename}: {metrics['scar_percentage']:.2f}% scar tissue detected")
        
        return {
            'filename': filename,
            'metrics': metrics,
            'prob_map_path': str(prob_path) if self.config['inference'].get('save_probability_maps', True) else None,
            'mask_path': str(mask_path),
            'visualization_path': str(vis_path) if self.config['inference'].get('save_visualizations', True) else None
        }
    
    def predict_batch(self, input_dir: str, output_dir: str) -> List[dict]:
        """Run inference on a batch of images."""
        # Create dataset
        dataset = ScarNetDataset(input_dir)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config['inference'].get('batch_size', 1),
            shuffle=False,
            num_workers=0
        )
        
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process images
        for batch in tqdm(dataloader, desc="Processing images"):
            images = batch['image'].to(self.device)
            paths = batch['path']
            filenames = batch['filename']
            
            with torch.no_grad():
                outputs = self.model(images)
            
            # Process each image in batch
            for i in range(len(images)):
                # Postprocess
                threshold = self.config['inference'].get('threshold', 0.5)
                prob_map, binary_mask = self.postprocess_output(outputs[i:i+1], threshold)
                
                # Calculate metrics
                metrics = self.calculate_metrics(prob_map, binary_mask)
                
                # Save results
                filename = Path(filenames[i]).stem
                
                # Save probability map
                if self.config['inference'].get('save_probability_maps', True):
                    prob_path = output_path / f"{filename}_probability.png"
                    cv2.imwrite(str(prob_path), (prob_map * 255).astype(np.uint8))
                
                # Save binary mask
                mask_path = output_path / f"{filename}_mask.png"
                cv2.imwrite(str(mask_path), binary_mask * 255)
                
                # Save metrics
                metrics_path = output_path / f"{filename}_metrics.yaml"
                with open(metrics_path, 'w') as f:
                    yaml.dump(metrics, f, default_flow_style=False)
                
                results.append({
                    'filename': filename,
                    'metrics': metrics,
                    'mask_path': str(mask_path)
                })
                
                logger.info(f"Processed {filename}: {metrics['scar_percentage']:.2f}% scar tissue detected")
        
        # Save summary
        summary_path = output_path / "summary.yaml"
        summary = {
            'total_images': len(results),
            'mean_scar_percentage': np.mean([r['metrics']['scar_percentage'] for r in results]),
            'std_scar_percentage': np.std([r['metrics']['scar_percentage'] for r in results]),
            'results': results
        }
        
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logger.info(f"Processed {len(results)} images. Summary saved to {summary_path}")
        
        return results


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="ScarNet Inference for Myocardial Scar Segmentation")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save results")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model weights")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to use for inference")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold for binary segmentation")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for inference")
    parser.add_argument("--single_image", type=str, default=None,
                       help="Process single image instead of directory")
    
    args = parser.parse_args()
    
    try:
        # Initialize inference
        scarnet = ScarNetInference(
            model_path=args.model_path,
            config_path=args.config,
            device=args.device
        )
        
        # Override config with command line arguments
        if args.threshold != 0.5:
            scarnet.config['inference']['threshold'] = args.threshold
        if args.batch_size != 1:
            scarnet.config['inference']['batch_size'] = args.batch_size
        
        # Run inference
        if args.single_image:
            logger.info(f"Processing single image: {args.single_image}")
            result = scarnet.predict_single(args.single_image, args.output_dir)
            print(f"\nResults for {result['filename']}:")
            print(f"  Scar percentage: {result['metrics']['scar_percentage']:.2f}%")
            print(f"  Scar area: {result['metrics']['scar_area_pixels']} pixels")
            print(f"  Mean probability: {result['metrics']['mean_scar_probability']:.3f}")
        else:
            logger.info(f"Processing directory: {args.input_dir}")
            results = scarnet.predict_batch(args.input_dir, args.output_dir)
            
            # Print summary
            scar_percentages = [r['metrics']['scar_percentage'] for r in results]
            print(f"\nProcessed {len(results)} images:")
            print(f"  Mean scar percentage: {np.mean(scar_percentages):.2f}% ± {np.std(scar_percentages):.2f}%")
            print(f"  Range: {np.min(scar_percentages):.2f}% - {np.max(scar_percentages):.2f}%")
        
        logger.info(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()