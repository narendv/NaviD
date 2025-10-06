import numpy as np
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple
import io
import lmdb
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from vint_train.data.data_utils import (
    img_path_to_data,
    get_data_path,
)


class DecoderDataset(Dataset):
    """
    Dataset for training decoder models that reconstruct RGB images from EfficientNet encodings.
    
    This dataset provides tuples of (encoding, target_image) where:
    - encoding: EfficientNet feature vector from the LMDB cache (shape: [1280])
    - target_image: Original RGB image that the encoding was derived from (shape: [3, H, W])
    """
    
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int] = (96, 96),
        min_sequence_length: int = 1,
        max_sequence_length: Optional[int] = None,
        normalize_images: bool = True,
        downscale_factor: int = 1,
    ):
        """
        Initialize the decoder dataset.
        
        Args:
            data_folder: Directory with all the image data
            data_split_folder: Directory with trajectory names and LMDB cache
            dataset_name: Name of the dataset
            image_size: Size of images to load as targets
            min_sequence_length: Minimum trajectory length to include
            max_sequence_length: Maximum trajectory length (None for no limit)
            normalize_images: Whether to normalize target images
            downscale_factor: Factor to downsample the dataset (for faster training/debugging)
        """
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.normalize_images = normalize_images
        
        # LMDB cache for encodings
        self._encoding_cache = None
        self._cache_path = os.path.join(self.data_split_folder, f"dataset_{self.dataset_name}_encodings.lmdb")
        
        # Load trajectory names
        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            self.traj_names = [line.strip() for line in f.readlines()]
        
        # Verify LMDB cache exists
        if not os.path.exists(self._cache_path):
            raise FileNotFoundError(
                f"Encoding cache not found at {self._cache_path}. "
                f"Please run preprocess_embeddings.py first to generate the cache."
            )
        
        # Build dataset index: list of (traj_name, timestep) tuples
        self.dataset_index = self._build_dataset_index()
        if downscale_factor > 1:
            self.dataset_index = self.dataset_index[::downscale_factor]
        
        print(f"DecoderDataset: Loaded {len(self.dataset_index)} image-encoding pairs from {len(self.traj_names)} trajectories")
    
    def _get_worker_cache(self):
        """Get or create LMDB cache for current worker process."""
        if self._encoding_cache is None:
            self._encoding_cache = lmdb.open(
                self._cache_path, readonly=True, lock=False, readahead=False, meminit=False
            )
        return self._encoding_cache
    
    def _build_dataset_index(self) -> List[Tuple[str, int]]:
        """Build index of all valid (traj_name, timestep) pairs."""
        dataset_index = []
        
        for traj_name in self.traj_names:
            try:
                traj_data = self._get_trajectory(traj_name)
                traj_len = len(traj_data["position"])
                
                # Skip trajectories that are too short
                if traj_len < self.min_sequence_length:
                    continue
                
                # Limit trajectory length if specified
                if self.max_sequence_length is not None:
                    traj_len = min(traj_len, self.max_sequence_length)
                
                # Add all valid timesteps
                for t in range(traj_len):
                    # Verify that this encoding exists in LMDB before adding to index
                    if self._encoding_exists(traj_name, t):
                        dataset_index.append((traj_name, t))
                    
            except Exception as e:
                print(f"Warning: Skipping trajectory {traj_name}: {e}")
                continue
        
        return dataset_index
    
    def _encoding_exists(self, traj_name: str, timestep: int) -> bool:
        """Check if encoding exists in LMDB cache."""
        key = f"{traj_name}_{timestep}".encode("utf-8")
        cache = self._get_worker_cache()
        
        with cache.begin(write=False) as txn:
            return txn.get(key) is not None
    
    def _get_trajectory(self, traj_name: str) -> Dict[str, Any]:
        """Load trajectory data from pickle file."""
        traj_path = os.path.join(self.data_folder, traj_name, "traj_data.pkl")
        with open(traj_path, "rb") as f:
            return pickle.load(f)
    
    def _get_cached_encoding(self, traj_name: str, timestep: int) -> torch.Tensor:
        """Get cached encoding from LMDB."""
        key = f"{traj_name}_{timestep}".encode("utf-8")
        
        # Get worker-specific cache connection
        cache = self._get_worker_cache()
        
        with cache.begin(write=False) as txn:
            buf = txn.get(key)
            if buf is None:
                raise KeyError(f"Key {traj_name}_{timestep} not found in LMDB cache")

            # Deserialize tensor and create a copy to avoid sharing issues
            embedding = torch.load(io.BytesIO(buf), map_location="cpu")
            # Create a new tensor to avoid multiprocessing sharing issues
            embedding = embedding.clone().detach().float()  # Convert from float16 to float32
            
        return embedding
    
    def _load_target_image(self, traj_name: str, timestep: int) -> torch.Tensor:
        """Load the target RGB image."""
        image_path = get_data_path(self.data_folder, traj_name, timestep)
        image_data = img_path_to_data(image_path, self.image_size)  # Returns [C, H, W] in [0, 1]
        
        # Skip normalization since we don't need it for decoder training
        # Images stay in [0, 1] range which is what LPIPS and SSIM expect
        
        return image_data
    
    def __len__(self) -> int:
        return len(self.dataset_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an encoding-image pair.
        
        Returns:
            Dictionary containing:
            - 'encoding': EfficientNet feature vector [1280] 
            - 'target_image': RGB image [3, H, W]
            - 'metadata': Additional information (trajectory name, timestep)
        """
        traj_name, timestep = self.dataset_index[idx]
        
        # Get the encoding from LMDB cache
        encoding = self._get_cached_encoding(traj_name, timestep)
        
        # Get the target image
        target_image = self._load_target_image(traj_name, timestep)
        
        # Create metadata
        metadata = {
            'traj_name': traj_name,
            'timestep': timestep,
            'dataset_name': self.dataset_name,
        }
        
        return {
            'encoding': encoding,
            'target_image': target_image,
            'metadata': metadata,
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # Example usage
    dataset = DecoderDataset(
        data_folder="/data/naren/recon",
        data_split_folder="/home/naren/NaviD/train/vint_train/data/data_splits/recon/test",
        dataset_name="recon",
        image_size=(96, 96),
        min_sequence_length=1,
        normalize_images=True,
        downscale_factor=10,  # Use every 10th sample for quick testing
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test single sample
    sample = dataset[0]
    print(f"Encoding shape: {sample['encoding'].shape}")
    print(f"Target image shape: {sample['target_image'].shape}")
    print(f"Target image range: [{sample['target_image'].min():.3f}, {sample['target_image'].max():.3f}]")
    print(f"Metadata: {sample['metadata']}")
    
    # Test dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    for batch in dataloader:
        print(f"Batch encoding shape: {batch['encoding'].shape}")
        print(f"Batch target_image shape: {batch['target_image'].shape}")
        break