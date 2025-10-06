#!/usr/bin/env python3
import os
import sys
import datetime
import torch
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import yaml
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from expnav.training.decoder_trainer import DecoderTrainer
from expnav.data.decoder_dataset import DecoderDataset
from easydict import EasyDict

# use medium precision for training
torch.set_float32_matmul_precision('medium')


def main(config):
    """Train the decoder model."""
    
    # Create data module
    train_dataset = DecoderDataset(
        data_folder=config.data_folder,
        data_split_folder=f"{config.data_split_folder}/train",
        dataset_name=config.dataset_name,
        image_size=config.image_size,
        min_sequence_length=config.min_sequence_length,
    )

    val_dataset = DecoderDataset(
        data_folder=config.data_folder,
        data_split_folder=f"{config.data_split_folder}/test",
        dataset_name=config.dataset_name,
        image_size=config.image_size,
        min_sequence_length=config.min_sequence_length,
        downscale_factor=10,  # Downsample validation set for speed
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )
    
    # Create model
    if config.pretrained:
        print(f"Loading pretrained model from {config.pretrained}")
        model = DecoderTrainer.load_from_checkpoint(
            checkpoint_path=config.pretrained,
            latent_n_channels=config.latent_n_channels,
            constant_size=config.constant_size,
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
            loss_type=config.loss_type,
            lambda1=config.lambda1,
            lambda2=config.lambda2,
            lambda3=config.lambda3,
        )

        # reconfigure optimizer with new lr and weight decay
        def _new_opt(self):
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=float(config.lr),
                weight_decay=float(config.weight_decay)
            )
            # scheduler = torch.optim.lr_scheduler.OneCycleLR(
            #     optimizer,
            #     max_lr=float(config.lr),
            #     total_steps=model.estimated_stepping_batches,
            #     pct_start=0.0,
            # )
            # return [optimizer], [scheduler]
            return [optimizer]
        model.configure_optimizers = _new_opt.__get__(model, DecoderTrainer)

    else:
        model = DecoderTrainer(
            latent_n_channels=config.latent_n_channels,
            gaussian_dim=config.gaussian_latent_dim,
            constant_size=config.constant_size,
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
            loss_type=config.loss_type,
            lambda1=config.lambda1,
            lambda2=config.lambda2,
            lambda3=config.lambda3,
        )
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Create logger
    if config.wandb:
        logger = WandbLogger(
            project=config.wandb_project,
            config=config,
            save_dir=config.output_dir,
            name=config.run_name
        )
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_total_loss",
        dirpath=os.path.join(config.output_dir, "checkpoints"),
        filename="decoder_model-{epoch:02d}-{val_total_loss:.4f}",
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
        # strategy="ddp_find_unused_parameters_true" if len(config.devices) > 1 else "auto",
        strategy="ddp" if len(config.devices) > 1 else "auto",
        precision="bf16-mixed"
    )
    
    # Train model
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # Test model
    trainer.test(model, val_dataloader)
    
    # Save final model
    trainer.save_checkpoint(os.path.join(config.output_dir, "checkpoints", "decoder_model_final.ckpt"))
    
    print(f"Training complete. Model saved to {config.output_dir}/checkpoints")


if __name__ == "__main__":
    with open("config/decoder.yaml", "r") as f:
        config = EasyDict(yaml.safe_load(f))

    # Create a subdirectory for this run (set date and time as run id)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = run_dir
    config.run_name = f"{run_id}"

    # save the config to the output directory
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(dict(config), f)

    print(f"Starting training")
        
    main(config)