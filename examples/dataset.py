import torch
import numpy as np
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from typing import List, Tuple, Any, Dict


class CustomCompose(Compose):
    def __call__(self, tensors: List[Any]) -> Tuple[Any, ...]:
        if not isinstance(tensors, (list, tuple)) or not tensors:
            return super().__call__(tensors)
        transformed_image = super().__call__(tensors[0])
        return tuple([transformed_image] + list(tensors[1:]))



class LeRobotWrapper(Dataset):

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
        self.action_window_size = action_window_size

        if visual_input and transform_list:
            filtered_transforms = [t for t in transform_list if t.__class__.__name__ != 'ToTensor']
            if len(filtered_transforms) < len(transform_list):
                print("[WARNING] Automatically removing redundant ToTensor from visual transforms.")
            transform_list = filtered_transforms

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
        actions = data[self.act_key].float()

        T_act = self.action_window_size  # (e.g., 16)
        D_act = actions.shape[-1]  # (e.g., 14)

        # 1. Check and squeeze the redundant dimension (LeRobotDataset often returns [T_act, 1, D_act])
        if actions.dim() == 3 and actions.shape[1] == 1:
            actions = actions.squeeze(1)  # Correctly yields [T_act, D_act]

        # Ensure the tensor is 2D at this point, if not, something is fundamentally wrong
        if actions.dim() != 2:
            # Added a more descriptive error message
            raise RuntimeError(
                f"Action tensor failure for item {idx}: Unexpected shape {actions.shape}. Expected 2D [T_act, D_act].")

        N_current_steps = actions.shape[0]

        # 2. Slice to the exact T_act length (kept for robust dataset handling)
        if N_current_steps != T_act:
            if N_current_steps > T_act:
                # Slice the end of the action sequence to get exactly T_act steps.
                actions = actions[-T_act:, :].clone().detach()

            # Re-check steps after slicing, allowing for potential undersize but raising error
            N_current_steps = actions.shape[0]

        # 3. Final validation
        if N_current_steps != T_act:
            raise RuntimeError(
                f"Action tensor failure for item {idx}. "
                f"Observed steps: {N_current_steps}. Expected steps: {T_act}."
            )

        mask = torch.ones(obs.shape[0], dtype=torch.bool)
        tensors = [obs, actions, mask]

        # 2. Apply transforms (to obs)
        if self.transform:
            tensors = self.transform(tensors)
            obs, actions, mask = tensors[0], tensors[1], tensors[2]

        goals = None
        if self.onehot_goals:
            task_idx = data["task_index"][0].long()
            onehot = F.one_hot(task_idx, num_classes=self.goal_dim).float()
            # Goals should be repeated for each observation in the window
            goals = onehot.unsqueeze(0).repeat(obs.shape[0], 1)

        # 4. Return a DICTIONARY
        output_dict = {
            "obs": obs,
            "target_actions": actions,  # Now guaranteed [T_act, D_act]
            "mask": mask,
        }

        if goals is not None:
            output_dict["goals"] = goals

        return output_dict


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
