import numpy as np
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple
import io
import lmdb
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torch.utils.data import get_worker_info
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from vint_train.data.data_utils import (
    img_path_to_data,
    get_data_path,
)
import warnings
from pydantic.warnings import UnsupportedFieldAttributeWarning
warnings.simplefilter("error", UnsupportedFieldAttributeWarning)
# then import/run your code



def lmdb_worker_init(_):
    """Worker init: ensure each worker starts with no inherited LMDB handle."""
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "_encoding_cache"):
        info.dataset._encoding_cache = None  # force lazy open within this worker


class AutoEncDataset(Dataset):
    """
    Dataset for training decoder models that reconstruct RGB images from EfficientNet encodings.

    Provides tuples of (encoding, target_image) where:
      - encoding: EfficientNet feature vector from the LMDB cache (shape: [1280])
      - target_image: Original RGB image (shape: [3, H, W], range [0,1])
    """

    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int] = (96, 96),
        downscale_factor: int = 1,
    ):
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        self.image_size = image_size

        # Load trajectory names
        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            self.traj_paths = [line.strip() for line in f.readlines()]
            self.traj_paths = [Path(data_folder) / name for name in self.traj_paths]
        
        # Get image paths
        self.rgb_files = []
        for traj in self.traj_paths:
            self.rgb_files.extend([f for f in traj.iterdir() if f.suffix == ".jpg"])

        if downscale_factor > 1:
            self.rgb_files = self.rgb_files[::downscale_factor]

        print(
            f"DecoderDataset: Loaded {len(self.rgb_files)} images "
            f"from {len(self.traj_paths)} trajectories"
        )

    # -------- I/O helpers --------
    def _load_target_image(self, traj_name: str, timestep: int) -> torch.Tensor:
        image_path = get_data_path(self.data_folder, traj_name, timestep)
        # Returns [C,H,W] in [0,1]
        image_data = img_path_to_data(image_path, self.image_size)
        return image_data

    # -------- Dataset API --------
    def __len__(self) -> int:
        return len(self.rgb_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        target_image = img_path_to_data(self.rgb_files[idx], self.image_size)

        return {
            "image": target_image,  # [3,H,W], [0,1]
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = AutoEncDataset(
        data_folder="/home/naren/recon",
        data_split_folder="/home/naren/NaviD/train/vint_train/data/data_splits/recon/test",
        dataset_name="recon",
        image_size=(96, 96),
        min_sequence_length=1,
        max_sequence_length=None,
        downscale_factor=10,
    )
    print(f"Dataset length: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=4,
    )
    for batch in dataloader:
        print(f"Batch image shape: {batch['image'].shape}")  # [B, 3, H, W]
        break