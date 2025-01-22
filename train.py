import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import logging
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

from scarnet.models.scarnet import ScarNet
from scarnet.data.dataset import CardiacDataset
from scarnet.utils.visualization import Visualizer
from scarnet.config import Config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ScarNet')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device ID')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--data_path', type=str, default='data', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    return parser.parse_args()

def compute_dice_score(pred, target, class_index):
    """Compute Dice score for a specific class."""
    pred_class = (pred == class_index).float()
    target_class = (target == class_index).float()
    
    intersection = (pred_class * target_class).sum()
    union = pred_class.sum() + target_class.sum()
    
    if union == 0:
        return 1.0  # If both prediction and target are empty, consider it perfect
    
    dice = (2. * intersection + 1e-8) / (union + 1e-8)
    return dice.item()

def train_epoch(model, loader, optimizer, scaler, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    dice_scores = {i: [] for i in range(4)}  # For 4 classes
    
    pbar = tqdm(loader, desc='Training')
    for batch_idx, data in enumerate(pbar):
        x, y = data['image'].to(device), data['mask'].to(device)

        # Mixed precision training
        with autocast(enabled=config.training.use_amp):
            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y.squeeze(1))

        # Scale loss and compute gradients
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config.training.accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Compute metrics
        pred = torch.argmax(out, dim=1)
        for class_idx in range(4):
            dice = compute_dice_score(pred, y.squeeze(1), class_idx)
            dice_scores[class_idx].append(dice)
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'scar_dice': np.mean(dice_scores[3])  # Class 3 is scar
        })

    return {
        'loss': total_loss / len(loader),
        'dice_scores': {k: np.mean(v) for k, v in dice_scores.items()}
    }

def validate(model, loader, device, visualizer=None, save_dir=None, epoch=None):
    """Validate the model."""
    model.eval()
    dice_scores = {i: [] for i in range(4)}
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader, desc='Validating')):
            x, y = data['image'].to(device), data['mask'].to(device)
            out = model(x)
            pred = torch.argmax(out, dim=1)
            
            for class_idx in range(4):
                dice = compute_dice_score(pred, y.squeeze(1), class_idx)
                dice_scores[class_idx].append(dice)
            
            # Save visualizations for first batch
            if batch_idx == 0 and visualizer and save_dir:
                visualizer.plot_segmentation(
                    x[0], y[0], pred[0],
                    save_path=os.path.join(save_dir, f'val_vis_epoch_{epoch}.png')
                )
    
    return {k: np.mean(v) for k, v in dice_scores.items()}

def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, save_path)

def main():
    args = parse_args()
    
    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    config = Config()
    if args.config:
        config.from_yaml(args.config)
    
    # Override config with command line arguments
    config.training.batch_size = args.batch_size
    config.training.epochs = args.epochs
    config.training.learning_rate = args.lr
    config.data.data_path = args.data_path
    
    # Setup logging and checkpoints
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(config.checkpoint_dir) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(save_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Initialize model
    model = ScarNet(
        pretrained_path=config.model.pretrained_path,
        num_classes=config.model.num_classes
    ).to(device)
    
    # Setup optimizer and scaler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    scaler = GradScaler(enabled=config.training.use_amp)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f'Resuming from epoch {start_epoch}')
    
    # Setup datasets
    train_dataset = CardiacDataset(
        x_files=sorted(Path(config.data.train_path).glob('**/Mag_image/*.h5')),
        y_files=[Path(str(x).replace('Mag_image', '4layer_mask')) for x in sorted(Path(config.data.train_path).glob('**/Mag_image/*.h5'))],
        imsize=config.data.image_size,
        augment=True,
        config=config
    )
    
    val_dataset = CardiacDataset(
        x_files=sorted(Path(config.data.test_path).glob('**/Mag_image/*.h5')),
        y_files=[Path(str(x).replace('Mag_image', '4layer_mask')) for x in sorted(Path(config.data.test_path).glob('**/Mag_image/*.h5'))],
        imsize=config.data.image_size,
        augment=False,
        config=config
    )
    
    # Setup dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    # Initialize visualizer
    visualizer = Visualizer()
    
    # Training loop
    best_scar_dice = 0.0
    for epoch in range(start_epoch, config.training.epochs):
        logging.info(f'\nEpoch {epoch+1}/{config.training.epochs}')
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, config)
        
        # Validate
        val_metrics = validate(
            model, val_loader, device, 
            visualizer=visualizer,
            save_dir=save_dir,
            epoch=epoch
        )
        
        # Log metrics
        logging.info(
            f'Train Loss: {train_metrics["loss"]:.4f}, '
            f'Train Scar Dice: {train_metrics["dice_scores"][3]:.4f}, '
            f'Val Scar Dice: {val_metrics[3]:.4f}'
        )
        
        # Save checkpoint
        save_checkpoint(
            model, optimizer, epoch,
            {'train': train_metrics, 'val': val_metrics},
            save_dir / f'checkpoint_epoch_{epoch+1}.pth'
        )
        
        # Save best model
        if val_metrics[3] > best_scar_dice:
            best_scar_dice = val_metrics[3]
            save_checkpoint(
                model, optimizer, epoch,
                {'train': train_metrics, 'val': val_metrics},
                save_dir / 'best_model.pth'
            )
            logging.info(f'New best model saved with Scar Dice: {best_scar_dice:.4f}')
        
        # Learning rate scheduling
        if hasattr(config.training, 'lr_schedule_factor'):
            lr = config.training.learning_rate * (
                config.training.lr_schedule_factor ** epoch
            )
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

if __name__ == '__main__':
    main()