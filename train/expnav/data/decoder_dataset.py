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


class DecoderDataset(Dataset):
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
        min_sequence_length: int = 1,
        max_sequence_length: Optional[int] = None,
        normalize_images: bool = True,   # kept for API compat; not used
        downscale_factor: int = 1,
    ):
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.normalize_images = normalize_images

        # LMDB cache path (env will be opened lazily per worker)
        self._encoding_cache = None
        self._cache_path = os.path.join(
            self.data_split_folder, f"dataset_{self.dataset_name}_encodings.lmdb"
        )

        # Load trajectory names
        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            self.traj_names = [line.strip() for line in f.readlines()]

        # Verify LMDB cache exists (but DO NOT open it here)
        if not os.path.exists(self._cache_path):
            raise FileNotFoundError(
                f"Encoding cache not found at {self._cache_path}. "
                f"Please run preprocess_embeddings.py first to generate the cache."
            )

        # Build dataset index WITHOUT leaving an env handle in self
        self.dataset_index = self._build_dataset_index_without_env()
        if downscale_factor > 1:
            self.dataset_index = self.dataset_index[::downscale_factor]

        print(
            f"DecoderDataset: Loaded {len(self.dataset_index)} image-encoding pairs "
            f"from {len(self.traj_names)} trajectories"
        )

    # -------- Fork-safety: don't serialize live env to workers --------
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_encoding_cache"] = None  # drop any open env before fork/pickle
        return state

    # -------- LMDB helpers --------
    def _get_worker_cache(self):
        """Open LMDB lazily per-worker on first access."""
        if self._encoding_cache is None:
            self._encoding_cache = lmdb.open(
                self._cache_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1024,
            )
        return self._encoding_cache

    def _build_dataset_index_without_env(self) -> List[Tuple[str, int]]:
        """
        Build list of (traj_name, timestep) pairs by pre-scanning LMDB keys
        with a short-lived env that is closed immediately (no handle kept).
        """
        dataset_index: List[Tuple[str, int]] = []

        # Pre-scan LMDB keys -> set of "traj_t" strings
        available: Optional[set] = set()
        try:
            env = lmdb.open(
                self._cache_path, readonly=True, lock=False, readahead=False, meminit=False
            )
            with env.begin(write=False) as txn:
                with txn.cursor() as cur:
                    for k, _ in cur:
                        # Keys were stored as f"{traj_name}_{t}".encode()
                        try:
                            available.add(k.decode("utf-8"))
                        except Exception:
                            # skip malformed keys
                            continue
            env.close()
        except Exception as e:
            print(f"Warning: could not pre-scan LMDB ({e}); proceeding without existence filter.")
            available = None  # fallback: don't filter by LMDB keys

        for traj_name in self.traj_names:
            try:
                traj_data = self._get_trajectory(traj_name)
                traj_len = len(traj_data["position"])

                if traj_len < self.min_sequence_length:
                    continue

                if self.max_sequence_length is not None:
                    traj_len = min(traj_len, self.max_sequence_length)

                for t in range(traj_len):
                    key = f"{traj_name}_{t}"
                    if (available is None) or (key in available):
                        dataset_index.append((traj_name, t))

            except Exception as e:
                print(f"Warning: Skipping trajectory {traj_name}: {e}")
                continue

        return dataset_index

    # -------- I/O helpers --------
    def _get_trajectory(self, traj_name: str) -> Dict[str, Any]:
        traj_path = os.path.join(self.data_folder, traj_name, "traj_data.pkl")
        with open(traj_path, "rb") as f:
            return pickle.load(f)

    def _get_cached_encoding(self, traj_name: str, timestep: int) -> torch.Tensor:
        key = f"{traj_name}_{timestep}".encode("utf-8")
        cache = self._get_worker_cache()
        with cache.begin(write=False) as txn:
            buf = txn.get(key)
            if buf is None:
                raise KeyError(f"Key {traj_name}_{timestep} not found in LMDB cache")
            embedding = torch.load(io.BytesIO(buf), map_location="cpu")
            embedding = embedding.clone().detach().float()  # ensure CPU float32 tensor
        return embedding

    def _load_target_image(self, traj_name: str, timestep: int) -> torch.Tensor:
        image_path = get_data_path(self.data_folder, traj_name, timestep)
        # Returns [C,H,W] in [0,1]
        image_data = img_path_to_data(image_path, self.image_size)
        return image_data

    # -------- Dataset API --------
    def __len__(self) -> int:
        return len(self.dataset_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_name, timestep = self.dataset_index[idx]
        encoding = self._get_cached_encoding(traj_name, timestep)
        target_image = self._load_target_image(traj_name, timestep)

        metadata = {
            "traj_name": traj_name,
            "timestep": timestep,
            "dataset_name": self.dataset_name,
        }

        return {
            "encoding": encoding,          # [1280]
            "target_image": target_image,  # [3,H,W], [0,1]
            "metadata": metadata,
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # Example usage
    dataset = DecoderDataset(
        data_folder="/home/naren/recon",
        data_split_folder="/home/naren/NaviD/train/vint_train/data/data_splits/recon/test",
        dataset_name="recon",
        image_size=(96, 96),
        min_sequence_length=1,
        normalize_images=True,
        downscale_factor=30,  # Use every 10th sample for quick testing
    )
    
    # Test dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=lmdb_worker_init,  # (2)
    )
    
    for batch in dataloader:
        print(f"Batch encoding shape: {batch['encoding'].shape}")
        print(f"Batch target_image shape: {batch['target_image'].shape}")
        # break