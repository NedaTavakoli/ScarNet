import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import logging

class Visualizer:
    """
    Visualization tools for cardiac MRI segmentation results.
    """
    def __init__(self):
        # Define custom colormaps
        self.seg_cmap = self._create_segmentation_colormap()
        
    def _create_segmentation_colormap(self):
        """Create custom colormap for segmentation visualization."""
        colors = [
            [0, 0, 0, 0],          # Background (transparent)
            [85/255, 85/255, 255/255, 0.7],   # Myocardium (blue)
            [255/255, 0, 127/255, 0.7],       # Blood pool (pink)
            [255/255, 255/255, 0, 0.7]        # Scar (yellow)
        ]
        return LinearSegmentedColormap.from_list('custom', colors, N=4)

    def plot_segmentation(self, image, mask, prediction=None, save_path=None):
        """
        Plot original image with ground truth and predicted segmentation.
        
        Args:
            image (torch.Tensor): Input image
            mask (torch.Tensor): Ground truth mask
            prediction (torch.Tensor, optional): Model prediction
            save_path (str, optional): Path to save the visualization
        """
        num_plots = 3 if prediction is not None else 2
        fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
        
        # Convert tensors to numpy
        image = image.cpu().numpy()
        mask = mask.cpu().numpy()
        if prediction is not None:
            prediction = prediction.cpu().numpy()
        
        # Plot original image
        axes[0].imshow(np.squeeze(image), cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Plot ground truth mask
        axes[1].imshow(np.squeeze(image), cmap='gray')
        axes[1].imshow(np.squeeze(mask), cmap=self.seg_cmap, alpha=0.5)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Plot prediction if available
        if prediction is not None:
            axes[2].imshow(np.squeeze(image), cmap='gray')
            axes[2].imshow(np.squeeze(prediction), cmap=self.seg_cmap, alpha=0.5)
            axes[2].set_title('Prediction')
            axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_attention_maps(self, image, attention_maps, save_path=None):
        """
        Visualize attention maps from the model.
        
        Args:
            image (torch.Tensor): Input image
            attention_maps (torch.Tensor): Attention maps from the model
            save_path (str, optional): Path to save the visualization
        """
        num_maps = attention_maps.shape[1]
        fig, axes = plt.subplots(1, num_maps + 1, figsize=(5*(num_maps + 1), 5))
        
        # Plot original image
        axes[0].imshow(np.squeeze(image.cpu().numpy()), cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        # Plot attention maps
        attention_maps = attention_maps.cpu().numpy()
        for i in range(num_maps):
            axes[i + 1].imshow(attention_maps[0, i], cmap='hot')
            axes[i + 1].set_title(f'Attention Map {i+1}')
            axes[i + 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_metrics(self, metrics_dict, save_path=None):
        """
        Plot training metrics history.
        
        Args:
            metrics_dict (dict): Dictionary containing metrics history
            save_path (str, optional): Path to save the plot
        """
        num_metrics = len(metrics_dict)
        fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 4))
        
        for idx, (metric_name, values) in enumerate(metrics_dict.items()):
            ax = axes[idx] if num_metrics > 1 else axes
            ax.plot(values)
            ax.set_title(metric_name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_comparison(self, images, predictions, ground_truth, save_path=None):
        """
        Plot comparison of predictions from different models.
        
        Args:
            images (list): List of input images
            predictions (dict): Dictionary of model predictions
            ground_truth (list): List of ground truth masks
            save_path (str, optional): Path to save the visualization
        """
        num_samples = len(images)
        num_models = len(predictions)
        
        fig, axes = plt.subplots(num_samples, num_models + 2, 
                                figsize=(3*(num_models + 2), 3*num_samples))
        
        for i in range(num_samples):
            # Plot original image
            axes[i, 0].imshow(np.squeeze(images[i].cpu().numpy()), cmap='gray')
            axes[i, 0].set_title('Input' if i == 0 else '')
            axes[i, 0].axis('off')
            
            # Plot ground truth
            axes[i, 1].imshow(np.squeeze(images[i].cpu().numpy()), cmap='gray')
            axes[i, 1].imshow(np.squeeze(ground_truth[i].cpu().numpy()), 
                            cmap=self.seg_cmap, alpha=0.5)
            axes[i, 1].set_title('Ground Truth' if i == 0 else '')
            axes[i, 1].axis('off')
            
            # Plot model predictions
            for j, (model_name, preds) in enumerate(predictions.items()):
                axes[i, j + 2].imshow(np.squeeze(images[i].cpu().numpy()), cmap='gray')
                axes[i, j + 2].imshow(np.squeeze(preds[i].cpu().numpy()), 
                                    cmap=self.seg_cmap, alpha=0.5)
                axes[i, j + 2].set_title(model_name if i == 0 else '')
                axes[i, j + 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def create_visualization_grid(self, results_dict, save_path=None):
        """
        Create a visualization grid showing different aspects of the model's output.
        
        Args:
            results_dict (dict): Dictionary containing different outputs to visualize
            save_path (str, optional): Path to save the visualization
        """
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 4)
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(np.squeeze(results_dict['image'].cpu().numpy()), cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Ground truth
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(np.squeeze(results_dict['image'].cpu().numpy()), cmap='gray')
        ax2.imshow(np.squeeze(results_dict['mask'].cpu().numpy()), 
                  cmap=self.seg_cmap, alpha=0.5)
        ax2.set_title('Ground Truth')
        ax2.axis('off')
        
        # Prediction
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(np.squeeze(results_dict['image'].cpu().numpy()), cmap='gray')
        ax3.imshow(np.squeeze(results_dict['prediction'].cpu().numpy()), 
                  cmap=self.seg_cmap, alpha=0.5)
        ax3.set_title('Prediction')
        ax3.axis('off')
        
        # Error map
        ax4 = fig.add_subplot(gs[0, 3])
        error = np.abs(results_dict['mask'].cpu().numpy() - 
                      results_dict['prediction'].cpu().numpy())
        ax4.imshow(np.squeeze(error), cmap='Reds')
        ax4.set_title('Error Map')
        ax4.axis('off')
        
        # Feature visualizations
        if 'features' in results_dict:
            features = results_dict['features']
            for i, (name, feat) in enumerate(features.items()):
                if i >= 6:  # Show max 6 feature maps
                    break
                ax = fig.add_subplot(gs[1 + i//3, i%3])
                ax.imshow(np.squeeze(feat[0].cpu().numpy()), cmap='viridis')
                ax.set_title(f'Feature: {name}')
                ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def save_attention_weights(attention_weights, save_path):
        """
        Save attention weights for later analysis.
        
        Args:
            attention_weights (torch.Tensor): Attention weights from the model
            save_path (str): Path to save the weights
        """
        try:
            np.save(save_path, attention_weights.cpu().numpy())
            logging.info(f"Attention weights saved to {save_path}")
        except Exception as e:
            logging.error(f"Error saving attention weights: {str(e)}")

# Utility functions for metrics visualization
def plot_dice_scores(dice_scores, save_path=None):
    """Plot Dice scores distribution."""
    plt.figure(figsize=(10, 6))
    plt.boxplot(dice_scores.values(), labels=dice_scores.keys())
    plt.title('Dice Scores Distribution')
    plt.ylabel('Dice Score')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(cm, classes, save_path=None):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()