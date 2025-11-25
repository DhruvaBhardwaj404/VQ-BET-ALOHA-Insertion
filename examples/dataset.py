import torch
import numpy as np
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from typing import List, Tuple, Any, Dict

# ==========================================================
# 1. CUSTOM TRANSFORM UTILITY 
# ==========================================================

class CustomCompose(Compose):
    """
    Custom wrapper to apply a sequence of torchvision transforms only to the 
    first element (image/observation) in the list returned by the Dataset.
    """
    def __call__(self, tensors: List[Any]) -> Tuple[Any, ...]:
        
        if not isinstance(tensors, (list, tuple)) or not tensors:
            return super().__call__(tensors)
            
        transformed_image = super().__call__(tensors[0])
        
        return tuple([transformed_image] + list(tensors[1:]))

# ==========================================================
# 2. DATASET WRAPPER
# ==========================================================

class LeRobotWrapper(Dataset):
    """
    A wrapper around LeRobotDataset to format data for VQ-BET training, 
    returning a dictionary with keys matching the training loop.
    """
    def __init__(
        self,
        repo_id: str,
        episodes: list[int],
        window_size: int,
        action_window_size: int,
        fps: int,
        visual_input: bool = True,
        onehot_goals: bool = False,
        goal_dim: int = 5,
        transform_list: List[Any] = None 
    ):
        self.onehot_goals = onehot_goals
        self.goal_dim = goal_dim
        self.transform = CustomCompose(transform_list) if transform_list else None
        
        self.obs_key = "observation.images.top" if visual_input else "observation.state"
        self.act_key = "action"

        dt = 1.0 / fps
        obs_deltas = [-1.0 * i * dt for i in range(window_size - 1, -1, -1)]
        act_deltas = [1.0 * i * dt for i in range(action_window_size)]

        delta_timestamps = {
            self.obs_key: obs_deltas,
            self.act_key: act_deltas
        }

        if self.onehot_goals:
            delta_timestamps["task_index"] = [0]

        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            episodes=episodes,
            delta_timestamps=delta_timestamps
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # 1. Fetch Data
        data = self.dataset[idx]
        
        obs = data[self.obs_key].float() 
        actions = data[self.act_key].float() # Shape: (T_action, D_action) -> e.g., (16, 14)
        
        mask = torch.ones(obs.shape[0], dtype=torch.bool)
        tensors = [obs, actions, mask]
        
        # 2. Apply transforms (to obs)
        if self.transform:
            tensors = self.transform(tensors)
            obs, actions, mask = tensors[0], tensors[1], tensors[2]
            
        T_act, D_act = actions.shape
        actions_reshaped = actions.view(T_act * D_act) # Flatten into a single vector.

        
        goals = None
        if self.onehot_goals:
            task_idx = data["task_index"][0].long() 
            onehot = F.one_hot(task_idx, num_classes=self.goal_dim).float()
            goals = onehot.unsqueeze(0).repeat(obs.shape[0], 1)
        
        # 4. Return a DICTIONARY 
        output_dict = {
            "obs": obs, # Shape: (T_obs, C, H, W) -> e.g., (16, 3, 224, 224)
            "target_actions": actions, # Shape: (T_action, D_action) -> e.g., (16, 14)
            "mask": mask, 
        }

        if goals is not None:
            output_dict["goals"] = goals

        return output_dict

# ==========================================================
# 3. DATA LOADING FUNCTION (Hydra Target)
# ==========================================================

def get_lerobot_train_val(
        repo_id: str,
        train_fraction: float = 0.9,
        window_size: int = 16,
        action_window_size: int = 16,
        visual_input: bool = True,
        onehot_goals: bool = False,
        goal_dim: int = 5,
        transform_list: List[Any] = None, 
        **kwargs 
):
    """Hydra target function to create train/val datasets."""
    meta = LeRobotDatasetMetadata(repo_id)
    total_episodes = meta.total_episodes
    fps = meta.fps
    
    all_episodes = np.arange(total_episodes)
    np.random.seed(42)
    np.random.shuffle(all_episodes)
    
    split_idx = int(total_episodes * train_fraction)
    train_episodes = all_episodes[:split_idx].tolist()
    val_episodes = all_episodes[split_idx:].tolist()

    common_args = {
        "repo_id": repo_id,
        "window_size": window_size,
        "action_window_size": action_window_size,
        "fps": fps,
        "visual_input": visual_input,
        "onehot_goals": onehot_goals,
        "goal_dim": goal_dim,
        "transform_list": transform_list,
    }
    
    train_ds = LeRobotWrapper(episodes=train_episodes, **common_args)
    val_ds = LeRobotWrapper(episodes=val_episodes, **common_args)

    return train_ds, val_ds
