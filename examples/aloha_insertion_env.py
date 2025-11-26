import logging
from typing import Callable, Dict, Optional, Tuple, Union, Any
import numpy as np
import torch
import gymnasium as gym
import gym_aloha
from gymnasium.envs.registration import register
# Assuming this is the correct path for your dataset loader
from dataset import LeRobotWrapper 
import einops
import torch.nn.functional as F 

# Setup logging
logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# --- Utility Function: Get Split Index ---
def get_split_idx(l, seed, train_fraction=0.95):
    """Utility function for splitting dataset indices."""
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(l, generator=rng).tolist()
    l_train = int(l * train_fraction)
    return idx[:l_train], idx[l_train:]

# --- Environment Wrapper: ALOHAWRAPPER ---

class ALOHAWRAPPER(gym.Wrapper):
    """
    Wrapper for Aloha Insertion. Implements the necessary output structure 
    to prevent the np.stack() crash during evaluation (goal_dim=0).
    """

    def __init__(self, id: str, goal_dim: int = 0, visual_input: bool = False, **kwargs):
        self.id = id
        self.visual_input = visual_input
        self.goal_dim = goal_dim 
        
        self.env = gym.make("gym_aloha/AlohaInsertion-v0", obs_type="pixels_agent_pos") 
        self.observation_space = self.env.observation_space 

        self.goal_masking = True
        self._goal_onehot: Optional[np.ndarray] = None
        self.tasks_to_complete = ["insertion"]

        super().__init__(self.env) 
        
        if self.visual_input and self.goal_dim == 0:
            logging.warning("ALOHAWRAPPER in FINAL FIX MODE: Returning (Tensor,) for np.stack safety.")

    def set_task_goal(self, one_hot_indices: torch.Tensor):
        self._goal_onehot = one_hot_indices.cpu().numpy() 

    def set_goal_masking(self, goal_masking=True):
        self.goal_masking = goal_masking
        
    def _preprocess_visual_obs(self, image_np: np.ndarray) -> torch.Tensor:
        """Converts raw image to processed PyTorch Tensor (C, H, W)."""
        # (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float() 
        # Resize/Interpolate to 224x224
        img_tensor = F.interpolate(img_tensor, size=224, mode='bilinear', align_corners=False).squeeze(0)
        return img_tensor 
            

    def _process_obs(self, obs_dict: Dict[str, Any]): 
        """Handles the CRITICAL MODE SWITCH."""
        state_vec = obs_dict.get('agent_pos', np.array([])).astype(np.float32)

        if self.visual_input:
            image_raw = obs_dict['pixels']['top']
            
            if self.goal_dim == 0:
                # EVALUATION MODE: Return a single-element tuple containing the tensor.
                # This ensures the output is one single item that np.stack can handle.
                processed_tensor = self._preprocess_visual_obs(image_raw)
                return (processed_tensor,) 
            
            else:
                # TRAINING MODE: Return the required raw NumPy tuple (image, state)
                return (image_raw.astype(np.uint8), state_vec)
        
        else:
            # State-Only Path
            return state_vec


    def reset(self, **kwargs) -> Tuple[Union[np.ndarray, Tuple[torch.Tensor], Tuple[np.ndarray, np.ndarray]], Dict]:
        # The base env returns (obs_dict, info) for gymnasium
        obs_dict, info = self.env.reset(**kwargs) 
        processed_obs = self._process_obs(obs_dict)
        self.num_achieved = 0

        # Ensure image is passed in the info dict
        if self.visual_input:
             info["image"] = obs_dict['pixels']['top']

        return processed_obs, info

    def step(self, action: np.ndarray) -> Tuple[Union[np.ndarray, Tuple[torch.Tensor], Tuple[np.ndarray, np.ndarray]], float, bool, Dict]:
        # The base env returns (obs_dict, reward, terminated, truncated, info) for gymnasium
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # Ensure image is passed in the info dict
        if self.visual_input:
            info["image"] = obs_dict['pixels']['top']
        
        processed_obs = self._process_obs(obs_dict)
            
        return processed_obs, reward, done, info

# --- Environment Registration ---
register(
    id="AlohaInsertion-eval-v0",
    entry_point="aloha_insertion_env:ALOHAWRAPPER",
    max_episode_steps=1200,
    reward_threshold=0.0,
)

# --- Goal Function (Designed for Training/Data-Loading) ---

def get_goal_fn(
    data_directory: str,
    goal_conditional: Optional[str] = None,
    goal_seq_len: Optional[int] = None,
    seed: Optional[int] = None,
    train_fraction: Optional[float] = None,
    unconditional: bool = False,
    goal_dim: int = 0,
    visual_input: bool = False,
) -> Callable[
    [gym.Env, torch.Tensor, torch.Tensor, torch.Tensor],
    Union[
        torch.Tensor,
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
        None,
    ],
]:
    """Retrieves goal information, returning raw tuple output when visual is True."""
    empty_tensor = torch.zeros(0)
    
    # Unconditional / Evaluation Case
    if unconditional or goal_dim == 0:
        return lambda env, state, goal_idx, frame_idx: (empty_tensor, {})

    # Goal-Conditioned Training Case
    relay_traj = LeRobotWrapper(
        repo_id=data_directory, episodes=None, window_size=1, 
        action_window_size=0, fps=1, onehot_goals=True, visual_input=visual_input
    )
    
    train_idx, val_idx = get_split_idx(
        len(relay_traj), seed=seed, train_fraction=train_fraction or 1.0,
    )
    
    static_onehot_goal = torch.zeros(goal_dim)
    if train_idx:
        example_data = relay_traj[train_idx[0]]
        if "goals" in example_data:
            static_onehot_goal = einops.reduce(example_data["goals"], "T C -> C", "max")
    
    goal_fn = lambda env, state, goal_idx, frame_idx: None 

    if goal_conditional == "future":
        assert goal_seq_len is not None
        goal_dataset_raw = relay_traj.dataset 
        
        def future_goal_fn(env, state, goal_idx, frame_idx): 
            info = {}
            if frame_idx == 0:
                env.set_task_goal(static_onehot_goal)
                info["onehot_goal"] = static_onehot_goal
            
            episode_index = train_idx[goal_idx]
            start_global_index = goal_dataset_raw.episodes_start_idx[episode_index]
            episode_length = goal_dataset_raw.episodes_len[episode_index]
            goal_start_global_index = start_global_index + episode_length - goal_seq_len
            data = goal_dataset_raw[goal_start_global_index]
            state_chunk = data["observation.state"].float()
            
            if visual_input:
                image_chunk = data.get("observation.images.top")
                if isinstance(image_chunk, dict): image_chunk = image_chunk.get("top")
                         
                # Return the raw NumPy data tuple (Image Chunk, State Chunk)
                goal_obs = (image_chunk.cpu().numpy(), state_chunk.cpu().numpy()) 
            else:
                goal_obs = state_chunk.cpu().numpy() 
            
            return goal_obs, info

        goal_fn = future_goal_fn

    elif goal_conditional == "onehot":

        def onehot_goal_fn(env, state, goal_idx, frame_idx):
            if frame_idx == 0:
                env.set_task_goal(static_onehot_goal)
            return static_onehot_goal, {}

        goal_fn = onehot_goal_fn

    else:
        raise ValueError(f"goal_conditional {goal_conditional} not recognized")

    return goal_fn
