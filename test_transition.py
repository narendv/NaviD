#!/usr/bin/env python3
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

from vint_train.training.transition_model_trainer import TransitionModelTrainer
from vint_train.data.transition_dataset import TransitionDataset
from easydict import EasyDict

# use medium precision for training
torch.set_float32_matmul_precision('medium')


def main(config):
    """Train the transition model using the NoMaD_ViNT encoder."""

    test_dataset = TransitionDataset(
                data_folder=config.data_folder,
                data_split_folder=f"{config.data_split_folder}/test",
                dataset_name=config.dataset_name,
                image_size=config.image_size,
                context_size=config.context_size,
                waypoint_spacing=config.waypoint_spacing,
                min_sequence_length=config.min_sequence_length,
                use_cache=True,
                downscale_factor=1,  # Downsample validation set for speed
            )

    test_dataloader = DataLoader(test_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=config.num_workers,
                                pin_memory=True,
                                drop_last=False,
                                persistent_workers=False,
                            )
    
    # Create model
    test_model_ckpt = './train/outputs/transition_model/checkpoints/transition_model_final.ckpt'
    model = TransitionModelTrainer.load_from_checkpoint(test_model_ckpt)
    model.eval()
    # model = torch.compile(model)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=config.max_epochs,
        log_every_n_steps=100,
        strategy="auto",
        precision="bf16-mixed"
    )
    
    # Test model
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    with open("train/config/forward.yaml", "r") as f:
        config = EasyDict(yaml.safe_load(f))
        
    main(config)