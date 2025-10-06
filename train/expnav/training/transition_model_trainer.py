import os
import sys
import argparse
from typing import Dict, Any, Optional, List, Tuple

from einops import rearrange
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.transforms.functional as TF

import numpy as np

# Import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from expnav.models.transition_model import TransitionModel
from efficientnet_pytorch import EfficientNet
from vint_train.models.nomad.nomad_vint import replace_bn_with_gn

# Ignore FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class TransitionModelTrainer(pl.LightningModule):
    """
    PyTorch Lightning module for training the transition model.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        action_dim: int = 2,
        num_attention_heads: int = 8,
        num_attention_layers: int = 4,
        ff_dim_factor: int = 4,
        dropout: float = 0.1,
        max_sequence_length: int = 10,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        use_cache: bool = True,
        encoder_checkpoint_path: Optional[str] = None,
        context_size: int = 3,
        image_size: Tuple[int, int] = (96, 96),
    ):
        """
        Initialize the transition model trainer.
        
        Args:
            d_model: dimension of the context vectors
            action_dim: Dimension of the action space
            num_attention_heads: Number of attention heads
            num_attention_layers: Number of transformer layers
            ff_dim_factor: Factor for feedforward dimension
            dropout: Dropout probability
            max_sequence_length: Maximum sequence length
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            use_cache: Whether to use cached encodings (True) or raw images (False)
            encoder_checkpoint_path: Path to encoder checkpoint (required if use_cache=False)
            context_size: Number of context frames (used if use_cache=False)
            image_size: Size of input images (used if use_cache=False)
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Store parameters
        self.use_cache = use_cache
        self.context_size = context_size
        self.image_size = image_size
        
        # Initialize encoder if not using cache
        self.encoder = None
        if not use_cache:
            if encoder_checkpoint_path is None:
                raise ValueError("encoder_checkpoint_path must be provided when use_cache=False")
            self.encoder = self._load_encoder(encoder_checkpoint_path)
        
        # Initialize the transition model
        self.transition_model = TransitionModel(
            d_model=d_model,
            action_dim=action_dim,
            num_attention_heads=num_attention_heads,
            num_attention_layers=num_attention_layers,
            ff_dim_factor=ff_dim_factor,
            dropout=dropout,
            max_sequence_length=max_sequence_length,
        )

        # Project states to a lower space
        # self.state_projection = nn.Linear(1280, d_model) \
        #     if d_model != 1280 else nn.Identity()
        
        # Loss function
        self.criterion = nn.L1Loss()
        
        # Learning parameters
        self.lr = lr
        self.weight_decay = weight_decay
    
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
    
    def _encode_raw_images(self, raw_images: torch.Tensor) -> torch.Tensor:
        """Encode raw images using the EfficientNet encoder."""
        if self.encoder is None:
            raise ValueError("Encoder not initialized")
        
        # Move encoder to same device as input
        if self.encoder.device != raw_images.device:
            self.encoder = self.encoder.to(raw_images.device)
        
        B, L = raw_images.shape[:2]
        raw_images = rearrange(raw_images, 'B L C H W -> (B F) C H W', L=L)
        
        with torch.no_grad():
            # Extract features
            obs_encoding = self.encoder.extract_features(raw_images)
            obs_encoding = self.encoder._avg_pooling(obs_encoding)
            
            if self.encoder._global_params.include_top:
                obs_encoding = obs_encoding.flatten(start_dim=1)
            
            # Reshape back to batch format: [B*(context_size+1), D] -> [B, (context_size+1)*D]
            obs_encoding = rearrange(obs_encoding, '(B F) D -> B F D', B=B, F=L)
        
        return obs_encoding
    
    def forward(self, curr_obs, action):
        """
        Forward pass through the transition model.
        
        Args:
            curr_obs: Current observation (either encoded features or raw images)
            action: Action taken
            
        Returns:
            Predicted next observation
        """
        # Encode raw images if not using cache
        if not self.use_cache:
            curr_obs = self._encode_raw_images(curr_obs)

        # curr_obs = self.state_projection(curr_obs)
        return self.transition_model(curr_obs, action)
    
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
        curr_obs = batch['curr_obs']
        action = batch['action']
        next_obs = batch['next_obs']
        
        # Forward pass
        predicted_next_obs = self(curr_obs, action)
        
        # Create temporal weights that emphasize the last observation
        seq_len = predicted_next_obs.shape[1]
        temporal_weights = torch.linspace(1.0, 3.0, seq_len, device=predicted_next_obs.device)
        temporal_weights /= temporal_weights.sum()
        temporal_weights = temporal_weights.view(1, -1, 1)  # Shape: [1, seq_len, 1]
        
        # Apply weights to predictions and targets
        weighted_predicted = predicted_next_obs * temporal_weights
        weighted_next_obs = next_obs * temporal_weights
        
        # Calculate weighted loss
        loss = self.criterion(weighted_predicted, weighted_next_obs)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return {'loss': loss}
    
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
        curr_obs = batch['curr_obs']
        action = batch['action']
        next_obs = batch['next_obs']
        
        # Forward pass
        predicted_next_obs = self(curr_obs, action)
        
        # Create temporal weights that emphasize the last observation
        seq_len = predicted_next_obs.shape[1]
        temporal_weights = torch.linspace(1.0, 3.0, seq_len, device=predicted_next_obs.device)
        temporal_weights /= temporal_weights.sum()
        temporal_weights = temporal_weights.view(1, -1, 1)  # Shape: [1, seq_len, 1]
        
        # Apply weights to predictions and targets
        weighted_predicted = predicted_next_obs * temporal_weights
        weighted_next_obs = next_obs * temporal_weights
        
        # Calculate weighted loss
        loss = self.criterion(weighted_predicted, weighted_next_obs)

        # Calculate additional metrics (unweighted for interpretability)
        mse = ((predicted_next_obs - next_obs) ** 2).mean(dim=(1, 2))
        mse_last = ((predicted_next_obs[:, -1] - next_obs[:, -1]) ** 2).mean(dim=-1)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mse', mse.mean(), on_step=False, on_epoch=True)
        self.log('val_mse_last', mse_last.mean(), on_step=False, on_epoch=True)
        
        return {'val_loss': loss, 'val_mse': mse}
    
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
        curr_obs = batch['curr_obs']
        action = batch['action']
        next_obs = batch['next_obs']
        
        # Forward pass
        predicted_next_obs = self(curr_obs, action)
        
        # Create temporal weights that emphasize the last observation
        seq_len = predicted_next_obs.shape[1]
        temporal_weights = torch.linspace(1.0, 3.0, seq_len, device=predicted_next_obs.device)
        temporal_weights /= temporal_weights.sum()
        temporal_weights = temporal_weights.view(1, -1, 1)  # Shape: [1, seq_len, 1]
        
        # Apply weights to predictions and targets
        weighted_predicted = predicted_next_obs * temporal_weights
        weighted_next_obs = next_obs * temporal_weights
        
        # Calculate weighted loss
        loss = self.criterion(weighted_predicted, weighted_next_obs)
        
        # Calculate additional metrics (unweighted for interpretability)
        mse = ((predicted_next_obs - next_obs) ** 2).mean(dim=(1, 2))
        mse_last = ((predicted_next_obs[:, -1] - next_obs[:, -1]) ** 2).mean(dim=-1)
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_mse', mse.mean(), on_step=False, on_epoch=True)
        self.log('test_mse_last', mse_last.mean(), on_step=False, on_epoch=True)
        
        return {'test_loss': loss, 'test_mse': mse}
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Optimizer and scheduler
        """
        # Get parameters to optimize (only transition model, encoder is frozen)
        parameters = list(self.transition_model.parameters())
        # Create optimizer
        optimizer = optim.AdamW(
            parameters,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Create learning rate scheduler with warmup
        scheduler = {
            'scheduler': OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start= 0.0,

            ),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

