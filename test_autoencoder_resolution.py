#!/usr/bin/env python3
"""
Script to test how changing input resolution affects AutoencoderKL output.
This script loads an example image, preprocesses it to different resolutions,
converts to [-1, 1] range, and uses the AutoencoderKL to reconstruct the images.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from diffusers.models import AutoencoderKL
import os


def preprocess_image(image_path: str, target_size: tuple) -> torch.Tensor:
    """
    Load and preprocess an image to tensor format in [-1, 1] range.
    
    Args:
        image_path: Path to the input image
        target_size: Target size as (height, width)
    
    Returns:
        Preprocessed tensor in [-1, 1] range with shape [1, 3, H, W]
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),  # Converts to [0, 1] range
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Converts to [-1, 1] range
    ])
    
    # Apply transforms and add batch dimension
    tensor = transform(image).unsqueeze(0)  # Shape: [1, 3, H, W]
    
    return tensor


def postprocess_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor from [-1, 1] range back to [0, 255] numpy array for visualization.
    
    Args:
        tensor: Input tensor in [-1, 1] range with shape [1, 3, H, W]
    
    Returns:
        Numpy array in [0, 255] range with shape [H, W, 3]
    """
    # Remove batch dimension and move to CPU
    tensor = tensor.squeeze(0).cpu()
    
    # Convert from [-1, 1] to [0, 1]
    tensor = (tensor + 1.0) / 2.0
    
    # Clamp to valid range
    tensor = torch.clamp(tensor, 0.0, 1.0)
    
    # Convert to numpy and transpose to HWC format
    array = tensor.permute(1, 2, 0).numpy()
    
    # Convert to [0, 255] range
    array = (array * 255).astype(np.uint8)
    
    return array


def test_autoencoder_resolution():
    """
    Test AutoencoderKL with different input resolutions.
    """
    # Initialize the AutoencoderKL
    print("Loading AutoencoderKL model...")
    autoencoder = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    autoencoder.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = autoencoder.to(device)
    print(f"Using device: {device}")
    
    # Test with different resolutions
    test_resolutions = [
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (96, 96),   # Your current training resolution
        (224, 224), # Common ImageNet resolution
    ]
    
    # Input image path - using the specific recon dataset image
    image_path = "/home/naren/recon/jackal_2019-08-15-17-28-56_8_r01/0.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found!")
        # Fallback to pipeline image if the specific image doesn't exist
        fallback_path = "/home/naren/NaviD/assets/pipeline.png"
        if os.path.exists(fallback_path):
            image_path = fallback_path
            print(f"Using fallback image: {fallback_path}")
        else:
            print("Creating a synthetic image instead.")
            # Create a synthetic test image
            synthetic_image = torch.randn(3, 256, 256)
            synthetic_image = (synthetic_image - synthetic_image.min()) / (synthetic_image.max() - synthetic_image.min())
            synthetic_image = transforms.ToPILImage()(synthetic_image)
            synthetic_image.save("synthetic_test_image.png")
            image_path = "synthetic_test_image.png"
    
    # Create output directory with specific name for this test
    output_dir = "recon_image_resolution_test"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Using recon dataset image: {image_path}")
    
    results = {}
    
    print(f"\nTesting AutoencoderKL with different resolutions:")
    print("=" * 60)
    
    with torch.no_grad():
        for i, (height, width) in enumerate(test_resolutions):
            print(f"\nTesting resolution: {height}x{width}")
            
            try:
                # Preprocess image
                input_tensor = preprocess_image(image_path, (height, width))
                input_tensor = input_tensor.to(device)
                
                print(f"Input tensor shape: {input_tensor.shape}")
                print(f"Input tensor range: [{input_tensor.min().item():.3f}, {input_tensor.max().item():.3f}]")
                
                # Encode to latent space
                latent = autoencoder.encode(input_tensor).latent_dist.sample()
                latent_shape = latent.shape
                
                print(f"Latent shape: {latent_shape}")
                print(f"Latent range: [{latent.min().item():.3f}, {latent.max().item():.3f}]")
                
                # Decode back to image space
                reconstructed = autoencoder.decode(latent).sample
                
                print(f"Reconstructed tensor shape: {reconstructed.shape}")
                print(f"Reconstructed tensor range: [{reconstructed.min().item():.3f}, {reconstructed.max().item():.3f}]")
                
                # Calculate reconstruction error
                mse_loss = F.mse_loss(input_tensor, reconstructed).item()
                l1_loss = F.l1_loss(input_tensor, reconstructed).item()
                
                print(f"Reconstruction MSE Loss: {mse_loss:.6f}")
                print(f"Reconstruction L1 Loss: {l1_loss:.6f}")
                
                # Store results
                results[f"{height}x{width}"] = {
                    'input_shape': input_tensor.shape,
                    'latent_shape': latent_shape,
                    'output_shape': reconstructed.shape,
                    'mse_loss': mse_loss,
                    'l1_loss': l1_loss,
                    'latent_spatial_size': (latent_shape[2], latent_shape[3]),
                    'compression_ratio': (height * width) / (latent_shape[2] * latent_shape[3] * latent_shape[1])
                }
                
                # Convert tensors to numpy for visualization
                input_np = postprocess_tensor(input_tensor)
                reconstructed_np = postprocess_tensor(reconstructed)
                
                # Create comparison visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                axes[0].imshow(input_np)
                axes[0].set_title(f"Original ({height}x{width})")
                axes[0].axis('off')
                
                # Reconstructed image
                axes[1].imshow(reconstructed_np)
                axes[1].set_title(f"Reconstructed\nMSE: {mse_loss:.6f}")
                axes[1].axis('off')
                
                # Difference map
                diff = np.abs(input_np.astype(float) - reconstructed_np.astype(float))
                axes[2].imshow(diff.astype(np.uint8))
                axes[2].set_title(f"Difference\nL1: {l1_loss:.6f}")
                axes[2].axis('off')
                
                plt.suptitle(f"AutoencoderKL Test - {height}x{width} Resolution\nRecon Dataset Image: jackal_2019-08-15-17-28-56_8_r01/0.jpg\nLatent: {latent_shape}")
                plt.tight_layout()
                
                # Save comparison
                output_path = os.path.join(output_dir, f"comparison_{height}x{width}.png")
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Save individual images
                Image.fromarray(input_np).save(os.path.join(output_dir, f"input_{height}x{width}.png"))
                Image.fromarray(reconstructed_np).save(os.path.join(output_dir, f"reconstructed_{height}x{width}.png"))
                
                print(f"Compression ratio: {results[f'{height}x{width}']['compression_ratio']:.2f}x")
                print(f"Saved results to {output_path}")
                
            except Exception as e:
                print(f"Error processing {height}x{width}: {str(e)}")
                continue
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Resolution':<12} {'Latent Shape':<15} {'MSE Loss':<12} {'L1 Loss':<12} {'Compression':<12}")
    print("-" * 80)
    
    for resolution, data in results.items():
        latent_shape_str = f"{data['latent_shape'][1]}x{data['latent_shape'][2]}x{data['latent_shape'][3]}"
        print(f"{resolution:<12} {latent_shape_str:<15} {data['mse_loss']:<12.6f} {data['l1_loss']:<12.6f} {data['compression_ratio']:<12.2f}x")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    if results:
        # Find best and worst reconstruction quality
        best_mse = min(results.values(), key=lambda x: x['mse_loss'])
        worst_mse = max(results.values(), key=lambda x: x['mse_loss'])
        
        best_res = [k for k, v in results.items() if v['mse_loss'] == best_mse['mse_loss']][0]
        worst_res = [k for k, v in results.items() if v['mse_loss'] == worst_mse['mse_loss']][0]
        
        print(f"Best reconstruction quality: {best_res} (MSE: {best_mse['mse_loss']:.6f})")
        print(f"Worst reconstruction quality: {worst_res} (MSE: {worst_mse['mse_loss']:.6f})")
        
        # Analyze latent space characteristics
        print(f"\nLatent space analysis:")
        for resolution, data in results.items():
            spatial_size = data['latent_spatial_size']
            print(f"  {resolution}: Latent spatial size {spatial_size[0]}x{spatial_size[1]}, "
                  f"Total latent elements: {data['latent_shape'][1] * data['latent_shape'][2] * data['latent_shape'][3]}")
        
        print(f"\nRecommendations:")
        print(f"- For your current training (96x96), latent spatial size is: {results.get('96x96', {}).get('latent_spatial_size', 'N/A')}")
        print(f"- Higher resolutions generally provide better reconstruction quality")
        print(f"- Consider the trade-off between quality and computational cost")
        print(f"- The AutoencoderKL compresses images by roughly 8x spatially (64x reduction in pixels)")
    
    print(f"\nResults saved in: {output_dir}/")
    return results


if __name__ == "__main__":
    print("AutoencoderKL Resolution Test")
    print("=" * 40)
    print("This script tests how different input resolutions affect")
    print("the AutoencoderKL's encoding and reconstruction quality.")
    print()
    
    results = test_autoencoder_resolution()
    
    print("\nTest completed! Check the generated images and analysis above.")