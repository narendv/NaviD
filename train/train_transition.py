#!/usr/bin/env python3
from json import encoder
import os
import sys
import torch
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import yaml
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from expnav.training.transition_model_trainer import TransitionModelTrainer
from expnav.data.transition_dataset import TransitionDataset
from easydict import EasyDict

# use medium precision for training
torch.set_float32_matmul_precision('medium')


def main(config):
    """Train the transition model using the NoMaD_ViNT encoder."""
    
    # No need to load encoder here - it's handled in the trainer if use_cache=False
    
    # Create data module
    use_cache = config.get('use_cache', True)
    train_dataset = TransitionDataset(
                data_folder=config.data_folder,
                data_split_folder=f"{config.data_split_folder}/train",
                dataset_name=config.dataset_name,
                image_size=config.image_size,
                context_size=config.context_size,
                waypoint_spacing=config.waypoint_spacing,
                min_sequence_length=config.min_sequence_length,
                use_cache=use_cache,
            )

    val_dataset = TransitionDataset(
                data_folder=config.data_folder,
                data_split_folder=f"{config.data_split_folder}/train",
                dataset_name=config.dataset_name,
                image_size=config.image_size,
                context_size=config.context_size,
                waypoint_spacing=config.waypoint_spacing,
                min_sequence_length=config.min_sequence_length,
                use_cache=use_cache,
                downscale_factor=10,  # Downsample validation set for speed
            )
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=False,
                            )
    
    val_dataloader = DataLoader(val_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=config.num_workers,
                                pin_memory=True,
                                drop_last=False,
                                persistent_workers=False,
                            )
    
    # Create model
    model = TransitionModelTrainer(
        d_model=config.d_model,
        action_dim=config.action_dim,
        num_attention_heads=config.num_attention_heads,
        num_attention_layers=config.num_attention_layers,
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
        use_cache=config.get('use_cache', True),
        encoder_checkpoint_path=config.enc_checkpoint if not config.get('use_cache', True) else None,
        context_size=config.context_size,
        image_size=config.image_size,
    )
    # model = torch.compile(model)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Create logger
    if config.wandb:
        logger = WandbLogger(
                project=config.wandb_project,
                config=config,
                # log_model="all" if args.wandb else False,
                save_dir=config.output_dir
            )
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(config.output_dir, "checkpoints"),
        filename="transition_model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config.devices,
        max_epochs=config.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger if config.wandb else None,
        log_every_n_steps=100,
        strategy="ddp_find_unused_parameters_true" if len(config.devices) > 1 else "auto",
        precision="bf16-mixed"
    )
    
    # Train model
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # Test model
    trainer.test(model, val_dataloader)
    
    # Save final model
    trainer.save_checkpoint(os.path.join(config.output_dir, "checkpoints", "transition_model_final.ckpt"))
    
    print(f"Training complete. Model saved to {config.output_dir}/checkpoints")


if __name__ == "__main__":
    with open("config/forward.yaml", "r") as f:
        config = EasyDict(yaml.safe_load(f))
        
    main(config)