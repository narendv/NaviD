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
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
)


class TransitionDataset(Dataset):
    """
    Dataset for training transition models that predict o_{t+1} from o_t and a_t.
    
    This dataset provides tuples of (observation_t, action_t, observation_{t+1})
    where observations are either encoded context vectors (if use_cache=True) 
    or raw context images (if use_cache=False).
    """
    
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        context_size: int = 3,
        context_type: str = "temporal",
        waypoint_spacing: int = 1,
        min_sequence_length: int = 10,
        max_sequence_length: Optional[int] = None,
        normalize: bool = True,
        learn_angle: bool = True,
        obs_type: str = "image",
        use_cache: bool = True,
        downscale_factor: int = 1,
    ):
        """
        Initialize the transition dataset.
        
        Args:
            data_folder: Directory with all the image data
            data_split_folder: Directory with trajectory names
            dataset_name: Name of the dataset
            image_size: Size of images to load
            context_size: Number of context frames
            context_type: Type of context ("temporal")
            waypoint_spacing: Spacing between waypoints
            min_sequence_length: Minimum trajectory length to include
            max_sequence_length: Maximum trajectory length (None for no limit)
            normalize: Whether to normalize images
            learn_angle: Whether to use sin/cos encoding for angles
            obs_type: Type of observations ("image")
            use_cache: Whether to use cached encodings (True) or raw images (False)
        """
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.context_size = context_size
        self.context_type = context_type
        self.waypoint_spacing = waypoint_spacing
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.normalize = normalize
        self.learn_angle = learn_angle
        self.obs_type = obs_type
        self.use_cache = use_cache
        
        # LMDB cache for encodings (only if using cache)
        self._encoding_cache = None
        
        # Load trajectory names
        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            self.traj_names = [line.strip() for line in f.readlines()]
        
        # Initialize LMDB cache if using cached encodings
        if self.use_cache:
            self._init_encoding_cache()
        
        # Build dataset index: list of (traj_name, timestep) tuples
        self.dataset_index = self._build_dataset_index()
        if downscale_factor > 1:
            self.dataset_index = self.dataset_index[::downscale_factor]

        
        print(f"TransitionDataset: Loaded {len(self.dataset_index)} transition pairs from {len(self.traj_names)} trajectories")
    
    def _init_encoding_cache(self):
        """Initialize LMDB cache for reading encoded observations."""
        self._cache_path = os.path.join(self.data_split_folder, f"dataset_{self.dataset_name}_encodings.lmdb")
        if not os.path.exists(self._cache_path):
            raise FileNotFoundError(
                f"Encoding cache not found at {self._cache_path}. "
                f"Please run preprocess_embeddings.py first to generate the cache."
            )
        # Don't open LMDB here - open it per worker process
        self._encoding_cache = None
        print(f"Cache path set to {self._cache_path}")
        
    def _get_worker_cache(self):
        """Get or create LMDB cache for current worker process."""
        if self._encoding_cache is None:
            self._encoding_cache = lmdb.open(
                self._cache_path, readonly=True, lock=False, readahead=False, meminit=False
            )
        return self._encoding_cache
    
    def _load_raw_context_images(self, traj_name: str, timestep: int) -> torch.Tensor:
        """Load raw context images."""
        if self.context_type == "temporal":
            context_times = list(
                range(
                    timestep - self.context_size * self.waypoint_spacing,
                    timestep + 1,
                    self.waypoint_spacing,
                )
            )
            context_images = []
            for t in context_times:
                image_path = get_data_path(self.data_folder, traj_name, t)
                image_data = img_path_to_data(image_path, self.image_size)
                if self.normalize:
                    image_data = TF.normalize(image_data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                context_images.append(image_data)
            
            return torch.stack(context_images)
        
    def _load_context_encodings(self, traj_name: str, timestep: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get the whole context (multiple frames) encodings"""
        if self.context_type == "temporal":
            context_times = list(
                range(
                    timestep - self.context_size * self.waypoint_spacing,
                    timestep + 1,
                    self.waypoint_spacing,
                )
            )
            context_enc = []
            for t in context_times:
                enc = self._get_cached_encoding(traj_name, t)
                context_enc.append(enc)
        return torch.stack(context_enc)
    
    def _build_dataset_index(self) -> List[Tuple[str, int]]:
        """Build index of all valid transition pairs."""
        dataset_index = []
        
        for traj_name in self.traj_names:
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])
            
            # Skip trajectories that are too short
            if traj_len < self.min_sequence_length:
                continue
            
            # Limit trajectory length if specified
            if self.max_sequence_length is not None:
                traj_len = min(traj_len, self.max_sequence_length)
            
            # Add valid timesteps (need context_size frames before and 1 frame after)
            valid_start = self.context_size * self.waypoint_spacing
            valid_end = traj_len - self.waypoint_spacing  # Need next timestep available
            
            for t in range(valid_start, valid_end, self.waypoint_spacing):
                dataset_index.append((traj_name, t))
        
        return dataset_index

    def _get_trajectory(self, traj_name: str) -> Dict[str, Any]:
        """Load trajectory data from pickle file."""
        traj_path = os.path.join(self.data_folder, traj_name, "traj_data.pkl")
        with open(traj_path, "rb") as f:
            return pickle.load(f)
    
    def _load_observation(self, traj_name: str, timestep: int) -> torch.Tensor:
        """Load observation (either from cache or raw images)."""
        if self.use_cache:
            return self._load_context_encodings(traj_name, timestep)
        else:
            # Load raw context images
            return self._load_raw_context_images(traj_name, timestep)

    def _get_cached_encoding(self, traj_name: str, timestep: int) -> Optional[torch.Tensor]:
        """Get cached encoding from LMDB if available."""
        key = f"{traj_name}_{timestep}".encode("utf-8")
        
        # Get worker-specific cache connection
        cache = self._get_worker_cache()
        
        with cache.begin(write=False) as txn:
            buf = txn.get(key)
            if buf is None:
                raise KeyError(f"Key {traj_name}_{timestep} not found")

            # Deserialize tensor and create a copy to avoid sharing issues
            embedding = torch.load(io.BytesIO(buf), map_location="cpu")
            # Create a new tensor to avoid multiprocessing sharing issues
            embedding = embedding.clone().detach()
        return embedding
    
    def __len__(self) -> int:
        return len(self.dataset_index)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a transition tuple (o_t, a_t, o_{t+1}, metadata).
        
        Returns:
            obs_t: Encoded observation at time t
            action_t: Action taken at time t
            obs_t_plus_1: Encoded observation at time t+1
            metadata: Additional information (trajectory name, timestep, etc.)
        """
        traj_name, timestep = self.dataset_index[idx]
        
        # Load trajectory data
        traj_data = self._get_trajectory(traj_name)
        
        # Get observations (either encoded or raw images)
        obs_t = self._load_observation(traj_name, timestep)
        obs_t_plus_1 = self._load_observation(traj_name, timestep + self.waypoint_spacing)
        
        # Extract position and orientation
        pos_t = np.array(traj_data["position"][timestep])
        pos_t_plus_1 = np.array(traj_data["position"][timestep + self.waypoint_spacing])
        
        yaw_t = traj_data["yaw"][timestep]
        yaw_t_plus_1 = traj_data["yaw"][timestep + self.waypoint_spacing]

        # Convert to local coordinates
        # to_local_coords returns a 2-element array; stack with dyaw as third column
        local_xy = to_local_coords(pos_t_plus_1, pos_t, yaw_t)  # shape (2,)
        dyaw = yaw_t_plus_1 - yaw_t  # scalar
        
        # Use linear and angular velocity as action
        linear_vel = np.sqrt(local_xy[0]**2 + local_xy[1]**2)
        angular_vel = dyaw
        action_t = np.array([linear_vel, angular_vel])
        
        # Convert to tensors
        action_t = torch.as_tensor(action_t, dtype=torch.float32)
        
        # # Apply angle encoding if requested
        # if self.learn_angle and len(action_t) > 1:
        #     action_t = calculate_sin_cos(action_t.unsqueeze(0)).squeeze(0)
        
        # Create metadata
        # metadata = {
        #     "traj_name": traj_name,
        #     "timestep": timestep,
        #     "dataset_name": self.dataset_name,
        # }
        metadata_tensor = torch.tensor([timestep], dtype=torch.int64)
        
        return {
            'curr_obs': obs_t,
            'action': action_t,
            'next_obs': obs_t_plus_1,
            'metadata': metadata_tensor,
        }
    
    def get_trajectory_sequence(self, traj_name: str, start_time: int, length: int) -> Dict[str, torch.Tensor]:
        """
        Get a sequence of transitions from a single trajectory.
        
        Args:
            traj_name: Name of trajectory
            start_time: Starting timestep
            length: Number of transitions to return
            
        Returns:
            Dictionary with observations, actions, and metadata
        """
        observations = []
        actions = []
        timesteps = []
        
        for i in range(length + 1):  # +1 because we need length+1 observations for length transitions
            timestep = start_time + i * self.waypoint_spacing
            
            # Get observation (encoded or raw images)
            obs = self._load_observation(traj_name, timestep)
            observations.append(obs)
            timesteps.append(timestep)
            
            # Get action (except for last observation)
            if i < length:
                try:
                    traj_data = self._get_trajectory(traj_name)
                    pos_t = np.array(traj_data["position"][timestep])
                    pos_t_plus_1 = np.array(traj_data["position"][timestep + self.waypoint_spacing])
                    yaw_t = traj_data["yaw"][timestep]
                    yaw_t_plus_1 = traj_data["yaw"][timestep + self.waypoint_spacing]
                    
                    dx = pos_t_plus_1[0] - pos_t[0]
                    dy = pos_t_plus_1[1] - pos_t[1]
                    dyaw = yaw_t_plus_1 - yaw_t
                    
                    action = np.array(
                        to_local_coords(pos_t_plus_1, pos_t, yaw_t),
                        yaw_t_plus_1 - yaw_t
                    )
                    
                    # Use linear and angular velocity as action
                    linear_vel = np.sqrt(action[0]**2 + action[1]**2)
                    angular_vel = action[2]
                    action_t = np.array([linear_vel, angular_vel])
                    
                except (KeyError, IndexError):
                    action_t = np.array([0.0, 0.0])
                
                action_t = torch.as_tensor(action_t, dtype=torch.float32)
                # if self.learn_angle:
                #     action_t = calculate_sin_cos(action_t.unsqueeze(0)).squeeze(0)
                
                actions.append(action_t)
        
        return {
            "observations": torch.stack(observations),
            "actions": torch.stack(actions),
            "timesteps": torch.tensor(timesteps),
            "traj_name": traj_name,
        }
    

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = TransitionDataset(
        data_folder="/data/naren/recon",
        data_split_folder="/home/naren/NaviD/train/vint_train/data/data_splits/recon/test",
        dataset_name="recon",
        image_size=(96, 96),
        context_size=3,
        waypoint_spacing=1,
        min_sequence_length=10,
        use_cache=True,
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    for batch in dataloader:
        print(f"Batch curr_obs shape: {batch['curr_obs'].shape}")
        print(f"Batch action shape: {batch['action'].shape}")
        print(f"Batch next_obs shape: {batch['next_obs'].shape}")
        print(f"Batch metadata shape: {batch['metadata'].shape}")
        break

