import os
import sys
import argparse
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.transforms.functional as TF

import numpy as np
import wandb
import lpips
from pytorch_msssim import ssim


# Import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# from expnav.models.style_decoder import StyleGanDecoder, RgbHead
from expnav.models.conv_decoder2 import ConvDecoder
# from expnav.models.beta_vae_decoder import BetaVAEDecoder, beta_vae_loss
from efficientnet_pytorch import EfficientNet
from vint_train.models.nomad.nomad_vint import replace_bn_with_gn

# Ignore FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class E2ETrainer(pl.LightningModule):
    """
    PyTorch Lightning module for training RGB decoder models.
    
    Takes EfficientNet encodings as input and reconstructs RGB images.
    """
    
    def __init__(
        self,
        encoder_checkpoint_path: str,
        latent_n_channels: int = 1280,  # EfficientNet-B0 output size
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        loss_type: str = "combined",  # "l1", "mse", or "combined"
        warmup_epochs: int = 10,
    ):
        """
        Initialize the decoder trainer.
        
        Args:
            latent_n_channels: Size of input latent vector (EfficientNet output)
            constant_size: Size of the initial constant tensor in StyleGAN decoder
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            loss_type: Type of loss function ("l1", "mse", or "combined")
            lambda1: Weight for L1 loss component
            lambda2: Weight for LPIPS loss component
            lambda3: Weight for SSIM loss component
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Store parameters
        self.latent_n_channels = latent_n_channels
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        self.warmup_epochs = warmup_epochs

        # Initialize encoder if not using cache
        self.encoder = None
        self.encoder = self._load_encoder(encoder_checkpoint_path)
        
        # # Initialize the decoder model
        # self.decoder = StyleGanDecoder(
        #     prediction_head=RgbHead,
        #     latent_n_channels=latent_n_channels,
        #     constant_size=constant_size,
        # )

        self.decoder = ConvDecoder(
            latent_n_channels=latent_n_channels,
            base_channels=512,
            # initial_spatial_size=3,
        )

        # self.decoder = BetaVAEDecoder(
        #     input_dim=latent_n_channels,
        #     latent_dim=gaussian_dim,
        #     loss_type='l1'
        # )
        self.decoder = torch.compile(self.decoder)
        
        # For logging validation images
        self.validation_outputs = []
        
        # Initialize loss functions
        self._init_loss_functions()
    
    def _init_loss_functions(self):
        """Initialize loss functions."""
        # L1 loss for pixel-level reconstruction
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # LPIPS loss for perceptual similarity
        if self.loss_type == "combined":
            self.lpips_loss = lpips.LPIPS(net='alex', verbose=False)
            # Freeze LPIPS parameters
            for param in self.lpips_loss.parameters():
                param.requires_grad = False
            print("LPIPS loss initialized")
        else:
            self.lpips_loss = None

    def _load_encoder(self, checkpoint_path: str) -> nn.Module:
        """Load and initialize the EfficientNet encoder."""
        # Create encoder
        encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3)
        encoder = replace_bn_with_gn(encoder)
        
        # Load checkpoint
        print(f"Loading encoder weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
        
        # Filter encoder parameters
        encoder_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('vision_encoder.obs_encoder'):
                new_key = k[len('vision_encoder.obs_encoder.'):]
                encoder_state_dict[new_key] = v
        
        # Load state dict
        encoder.load_state_dict(encoder_state_dict, strict=True)
        print("Successfully loaded encoder weights.")
        
        # Set to eval mode (we don't want to train the encoder)
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        
        return encoder
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            encoding: EfficientNet encoding [B, 1280]
            
        Returns:
            Reconstructed RGB image [B, 3, H, W]
        """
        encodings = self.encoder.extract_endpoints(images)
        # feats = encodings['reduction_6']  # [B, 1280, 3, 3]
        # feats = self.encoder._avg_pooling(feats)
        # feats = torch.flatten(feats, 1)
        # encodings = self.decoder(feats, encodings)

        feats = encodings['reduction_4']  # [B, 1280, 3, 3]
        encodings = self.decoder(feats)
        return encodings
    
    def _compute_loss(self, pred_outputs: Dict[str, torch.Tensor], target_image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction loss using combined loss function:
        L = λ1 * L1 + λ2 * LPIPS + λ3 * (1 - SSIM)
        
        Args:
            pred_image: Predicted RGB image [B, 3, H, W] in [-1, 1] (Tanh output) or [0, 1]
            target_image: Target RGB image [B, 3, H, W] in [0, 1]
            
        Returns:
            Dictionary with loss components
        """
        losses = {}
        pred_image_raw = pred_outputs['rgb_2']
        
        # Normalize Tanh output to [0, 1] range to match target
        pred_image = (pred_image_raw + 1.0) / 2.0  # Convert [-1, 1] to [0, 1]
        if self.loss_type == "combined":
            # L1 loss component
            l1_loss = self.l1_loss(pred_image, target_image)
            losses['l1_loss'] = l1_loss
            if self.current_epoch < 10:
                # Warm-up phase: only use L1 loss for first 10 epochs
                losses['total_loss'] = l1_loss
                return losses
            
            # Initialize total loss with L1 component
            total_loss = 1.0 * l1_loss
            
            # LPIPS loss component
            # LPIPS expects images in [-1, 1] range
            # Scale from [0, 1] to [-1, 1]
            pred_lpips = pred_image * 2.0 - 1.0
            target_lpips = target_image * 2.0 - 1.0
            
            lpips_loss = self.lpips_loss(pred_lpips, target_lpips).mean()
            losses['lpips_loss'] = lpips_loss
            total_loss += 0.3 * lpips_loss
            
            # SSIM loss component
            ssim_value = ssim(pred_image, target_image, data_range=1.0, size_average=True)
            ssim_loss = 1.0 - ssim_value  # Convert to loss (higher SSIM = lower loss)
            losses['ssim_loss'] = ssim_loss
            total_loss += 0.2 * ssim_loss
            
            losses['total_loss'] = total_loss
            
        elif self.loss_type == "l1":
            l1_loss = self.l1_loss(pred_image, target_image)
            # losses['l1_loss'] = l1_loss
            losses['total_loss'] = l1_loss
            
        elif self.loss_type == "elbo":
            beta = 0.5 if self.current_epoch >= self.warmup_epochs else 0.0
            mu = pred_outputs['mu']
            log_var = pred_outputs['log_var']
            total_loss, rec, kl = beta_vae_loss(pred_image, target_image, mu, log_var, beta=beta, recon_type="l1")
            losses['total_loss'] = total_loss
            losses['l1_loss'] = rec
            losses['kl_loss'] = kl
        
        return losses
    
    def _log_images_to_wandb(self, pred_images: torch.Tensor, target_images: torch.Tensor, prefix: str = "val"):
        """
        Log comparison images to wandb.
        
        Args:
            pred_images: Predicted images [B, 3, H, W] in [0, 1]
            target_images: Target images [B, 3, H, W] in [0, 1]  
            prefix: Prefix for wandb log keys
        """
        if not isinstance(self.logger, pl.loggers.WandbLogger):
            return
        # Convert to numpy and transpose for wandb (expects HWC format)
        pred_np = pred_images.float().numpy().transpose(0, 2, 3, 1)  # [B, H, W, 3]
        target_np = target_images.float().numpy().transpose(0, 2, 3, 1)  # [B, H, W, 3]

        # scale back to [0, 255] and convert to uint8
        pred_np = (pred_np * 255).astype(np.uint8)
        target_np = (target_np * 255).astype(np.uint8)

        # Create wandb images
        wandb_images = []
        for i in range(2):  # Log first 2 images
            # Create side-by-side comparison
            pred_img = pred_np[i]
            target_img = target_np[i]
            
            # Concatenate horizontally (target | predicted)
            comparison = np.concatenate([target_img, pred_img], axis=1)
            
            wandb_images.append(
                wandb.Image(
                    comparison, 
                    caption=f"Left: Ground Truth | Right: Predicted (Sample {i+1})"
                )
            )
        
        # Log to wandb with epoch number
        self.logger.experiment.log({
            f"{prefix}_outputs": wandb_images
        }, step=self.global_step)
    
    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Loss dictionary
        """
        # Get batch data
        input_image = batch['image']  # [B, 3, H, W]
        
        # Forward pass
        pred_outputs = self(input_image)
        
        # Compute loss
        losses = self._compute_loss(pred_outputs, input_image)

        # Log metrics
        for loss_name, loss_value in losses.items():
            self.log(f'train_{loss_name}', loss_value, on_step=True, on_epoch=False, prog_bar=(loss_name == 'total_loss'), sync_dist=True)
        
        return losses['total_loss']
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Loss dictionary
        """
        # Get batch data
        input_image = batch['image']  # [B, 3, H, W]
        
        # Forward pass
        pred_outputs = self(input_image)
        
        # Compute loss
        losses = self._compute_loss(pred_outputs, input_image)
        pred_image = (pred_outputs['rgb_2'] + 1.0) / 2.0  # Convert [-1, 1] to [0, 1]
        if 'l1' not in losses:
            l1_loss = self.l1_loss(pred_image, input_image)
            losses['l1_loss'] = l1_loss
        
        # # Calculate additional metrics
        # with torch.no_grad():
        #     # PSNR (Peak Signal-to-Noise Ratio)
        #     mse = F.mse_loss(pred_image, target_image)
        #     psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
            
        #     # Cosine similarity between flattened images
        #     pred_flat = pred_image.view(pred_image.size(0), -1)
        #     target_flat = target_image.view(target_image.size(0), -1)
        #     correlation = F.cosine_similarity(pred_flat, target_flat, dim=1).mean()
        
        # Log metrics
        for loss_name, loss_value in losses.items():
            self.log(f'val_{loss_name}', loss_value, on_step=False, on_epoch=True, prog_bar=(loss_name == 'total_loss'), sync_dist=True)

        # self.log('val_psnr', psnr, on_step=False, on_epoch=True, sync_dist=True)
        # self.log('val_correlation', correlation, on_step=False, on_epoch=True, sync_dist=True)
        
        # Log images to wandb (only for first batch to avoid spam)
        if batch_idx == 0:
            with torch.no_grad():
                self.validation_outputs = {
                    'pred_image': pred_image[:2].clone().detach().cpu(),
                    'target_image': input_image[:2].clone().detach().cpu()
                }

        return {
            'val_loss': losses['total_loss'],
            # 'val_psnr': psnr,
            # 'val_correlation': correlation,
            # 'pred_image': pred_image[:4],  # Save first 4 images for logging
            # 'target_image': target_image[:4],
        }
    
    def on_validation_epoch_end(self):
        """Log images at the end of validation epoch."""
        if self.validation_outputs:
            self._log_images_to_wandb(
                self.validation_outputs['pred_image'],
                self.validation_outputs['target_image'],
                prefix="val"
            )
            # Clear outputs
            self.validation_outputs = []

    def test_step(self, batch, batch_idx):
        """
        Test step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Loss dictionary
        """
        # Get batch data
        input_image = batch['image']  # [B, 3, H, W]
        
        # Forward pass
        pred_outputs = self(input_image)
        
        # Compute loss
        losses = self._compute_loss(pred_outputs, input_image)

        # Calculate additional metrics
        with torch.no_grad():
            mse = F.mse_loss(pred_outputs['rgb_2'], input_image)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
            
            # pred_flat = pred_image.view(pred_image.size(0), -1)
            # target_flat = target_image.view(target_image.size(0), -1)
            # correlation = F.cosine_similarity(pred_flat, target_flat, dim=1).mean()
        
        # Log metrics
        for loss_name, loss_value in losses.items():
            self.log(f'test_{loss_name}', loss_value, on_step=False, on_epoch=True)
        
        self.log('test_psnr', psnr, on_step=False, on_epoch=True)
        # self.log('test_correlation', correlation, on_step=False, on_epoch=True)
        
        return {
            'test_loss': losses['total_loss'],
            'test_psnr': psnr,
            # 'test_correlation': correlation,
        }
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Optimizer and scheduler
        """
        # Create optimizer
        optimizer = optim.AdamW(
            self.decoder.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Create learning rate scheduler
        scheduler = {
            'scheduler': OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.2,
            ),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        """Configure gradient clipping to prevent exploding gradients."""
        self.clip_gradients(
            optimizer, 
            gradient_clip_val=1.0,  # Clip gradients to max norm of 1.0
            gradient_clip_algorithm="norm"
        )

if __name__ == "__main__":
    
    # Create model
    model = E2ETrainer(
        encoder_checkpoint_path='/home/naren/NaviD/checkpoints/nomad.pth',
        latent_n_channels=112,
    )
    
    input = torch.randn((2, 3, 96, 96))
    out = model(input)
    for k, v in out.items():
        print(f"{k}: {v.shape}")