#!/usr/bin/env python3
"""
Visualization script for trained decoder model.
Loads a trained decoder model and generates example GT vs predicted image comparisons.
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add the training modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'train'))
from expnav.training.decoder_trainer import DecoderTrainer
from expnav.data.decoder_dataset import DecoderDataset


def create_comparison_grid(gt_images, pred_images, max_samples=4):
    """
    Create a grid showing GT vs Predicted images side by side.
    
    Args:
        gt_images: Ground truth images [B, 3, H, W] 
        pred_images: Predicted images [B, 3, H, W]
        max_samples: Maximum number of samples to show
        
    Returns:
        PIL Image of the comparison grid
    """
    batch_size = min(gt_images.size(0), max_samples)
    
    # Convert tensors to numpy arrays [0, 1] range
    gt_np = gt_images[:batch_size].detach().cpu().numpy()
    pred_np = pred_images[:batch_size].detach().cpu().numpy()
    
    # Transpose from [B, C, H, W] to [B, H, W, C] and clip to [0, 1]
    gt_np = np.transpose(gt_np, (0, 2, 3, 1))
    pred_np = np.transpose(pred_np, (0, 2, 3, 1))
    gt_np = np.clip(gt_np, 0, 1)
    pred_np = np.clip(pred_np, 0, 1)
    
    # Create figure
    fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 4, 8))
    if batch_size == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(batch_size):
        # Ground truth images in top row
        axes[0, i].imshow(gt_np[i])
        axes[0, i].set_title(f'Ground Truth {i+1}', fontsize=12)
        axes[0, i].axis('off')
        
        # Predicted images in bottom row
        axes[1, i].imshow(pred_np[i])
        axes[1, i].set_title(f'Predicted {i+1}', fontsize=12)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Convert to PIL Image
    fig.canvas.draw()
    # Use the updated matplotlib API
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)
    # Convert RGBA to RGB
    img_array = img_array[:, :, :3]
    plt.close(fig)
    
    return Image.fromarray(img_array)


def create_side_by_side_comparison(gt_images, pred_images, max_samples=4):
    """
    Create side-by-side comparison images (GT | Pred) for each sample.
    
    Args:
        gt_images: Ground truth images [B, 3, H, W]
        pred_images: Predicted images [B, 3, H, W]
        max_samples: Maximum number of samples to create
        
    Returns:
        List of PIL Images with side-by-side comparisons
    """
    batch_size = min(gt_images.size(0), max_samples)
    comparison_images = []
    
    for i in range(batch_size):
        # Get single images and convert to numpy
        gt_img = gt_images[i].detach().cpu().numpy()
        pred_img = pred_images[i].detach().cpu().numpy()
        
        # Transpose from [C, H, W] to [H, W, C] and clip to [0, 1]
        gt_img = np.transpose(gt_img, (1, 2, 0))
        pred_img = np.transpose(pred_img, (1, 2, 0))
        gt_img = np.clip(gt_img, 0, 1)
        pred_img = np.clip(pred_img, 0, 1)
        
        # Convert to uint8
        gt_img = (gt_img * 255).astype(np.uint8)
        pred_img = (pred_img * 255).astype(np.uint8)
        
        # Convert to PIL Images
        gt_pil = Image.fromarray(gt_img)
        pred_pil = Image.fromarray(pred_img)
        
        # Create side-by-side image
        width, height = gt_pil.size
        comparison = Image.new('RGB', (width * 2, height))
        comparison.paste(gt_pil, (0, 0))
        comparison.paste(pred_pil, (width, 0))
        
        comparison_images.append(comparison)
    
    return comparison_images


def load_model_and_generate_examples(
    checkpoint_path: str,
    data_dir: str,
    data_split_folder: str,
    output_dir: str = "./decoder_examples",
    num_samples: int = 4,
    device: str = "auto"
):
    """
    Load trained decoder model and generate example comparisons.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        data_dir: Path to dataset directory
        output_dir: Directory to save comparison images
        num_samples: Number of examples to generate
        device: Device to use ("auto", "cuda", "cpu")
    """
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model from checkpoint
    print(f"Loading model from: {checkpoint_path}")
    model = DecoderTrainer.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    
    # Create dataset and dataloader
    print(f"Loading dataset from: {data_dir}")
    dataset = DecoderDataset(
        data_folder=data_dir,
        data_split_folder=data_split_folder,
        dataset_name="recon",
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=num_samples,
        shuffle=True,
        num_workers=4,
        pin_memory=device == "cuda"
    )
    
    # Generate examples
    print("Generating examples...")
    with torch.no_grad():
        # Get a batch of data
        batch = next(iter(dataloader))
        encodings = batch['encoding'].to(device)
        gt_images = batch['target_image'].to(device)
        
        # Generate predictions
        pred_images = model(encodings)
        
        # Calculate loss for reference
        losses = model._compute_loss(pred_images, gt_images)
        print(f"Example batch losses:")
        for loss_name, loss_value in losses.items():
            print(f"  {loss_name}: {loss_value:.4f}")
        
        # Create comparison grid
        grid_image = create_comparison_grid(gt_images, pred_images, num_samples)
        grid_path = output_dir / "decoder_comparison_grid.png"
        grid_image.save(grid_path)
        print(f"Saved comparison grid: {grid_path}")
        
        # Create individual side-by-side comparisons
        side_by_side_images = create_side_by_side_comparison(gt_images, pred_images, num_samples)
        for i, img in enumerate(side_by_side_images):
            img_path = output_dir / f"decoder_comparison_{i+1}.png"
            img.save(img_path)
            print(f"Saved comparison {i+1}: {img_path}")
    
    print(f"\nAll examples saved to: {output_dir}")
    print("Files created:")
    print("  - decoder_comparison_grid.png (2x4 grid showing GT vs Pred)")
    print("  - decoder_comparison_N.png (individual side-by-side comparisons)")


def main():
    parser = argparse.ArgumentParser(description="Visualize trained decoder model")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to trained model checkpoint (.ckpt file)"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="/data/naren/recon",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--data_split_folder", 
        type=str, 
        default="/home/naren/NaviD/train/vint_train/data/data_splits/recon/test",
        help="Path to data split folder"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./decoder_examples",
        help="Directory to save example images (default: ./decoder_examples)"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=4,
        help="Number of example images to generate (default: 4)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        return
    
    # Generate examples
    load_model_and_generate_examples(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        data_split_folder=args.data_split_folder,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        device=args.device
    )


if __name__ == "__main__":
    main()