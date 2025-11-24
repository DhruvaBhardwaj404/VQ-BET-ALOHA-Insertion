import abc
from typing import Any, Callable, List, Optional, Sequence, Dict, Union
import numpy as np
import torch
import itertools  # ADDED: Use standard Python itertools for accumulation
from torch import default_generator, randperm
# REMOVED: from torch._utils import _accumulate (Causes ImportError)
from torch.utils.data import Dataset, Subset
import einops
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# --- ABSTRACT BASE CLASS ---

class TrajectoryDataset(Dataset, abc.ABC):
    """
    A dataset containing whole trajectories (episodes).
    TrajectoryDataset[i] returns: (observations, actions, mask, [goals])
    """

    @abc.abstractmethod
    def get_seq_length(self, idx):
        """Returns the length of the idx-th trajectory."""
        raise NotImplementedError


# --- SUBSET CLASS ---

class TrajectorySubset(TrajectoryDataset, Subset):
    """Subset of a trajectory dataset at specified indices."""

    def __init__(self, dataset: TrajectoryDataset, indices: Sequence[int]):
        Subset.__init__(self, dataset, indices)

    def get_seq_length(self, idx):
        # We rely on the underlying TrajectoryDataset to provide the length
        return self.dataset.get_seq_length(self.indices[idx])


class InsertionAlohaTrajectoryDataset(TrajectoryDataset):
    """
    A Dataset class wrapping LeRobotDataset to expose whole trajectories (episodes),
    matching the TrajectoryDataset API required by TrajectorySlicerDataset.

    This class handles the episode-based splitting and ensures data is fetched
    and aggregated across the entire episode.
    """

    def __init__(
            self,
            repo_id: str,
            split: str,
            train_fraction: float = 0.9,
            device: str = "cuda",
            onehot_goals: bool = False,
            visual_input: bool = True,  # Default to True for insertion task
            goal_dim: int = 5,
            **kwargs
    ):
        self.device = torch.device(device)
        self.split = split
        self.onehot_goals = onehot_goals
        self.visual_input = visual_input
        self.GOAL_DIM = goal_dim

        # 1. Load the full LeRobotDataset
        print(f"LeRobot: Attempting to load dataset from repo_id: {repo_id}")
        self.full_dataset = LeRobotDataset(repo_id=repo_id, **kwargs)
        print(f"LeRobot: Dataset object created successfully. Total samples: {len(self.full_dataset)}")

        # 2. Get the Episode Data (MANUALLY computed from hf_dataset for robustness)

        # Use the underlying Hugging Face Dataset (hf_dataset)
        hf_ds = self.full_dataset.hf_dataset

        if 'episode_index' not in hf_ds.column_names:
            raise RuntimeError(
                f"--- LeRobot Dataset Error ---\n"
                f"Could not find 'episode_index' column in the dataset, required for trajectory splitting.\n"
                f"-----------------------------"
            )

        episode_indices = hf_ds['episode_index']

        self.episode_metadata = {}
        current_episode_id = -1
        start_frame = -1

        for i, episode_id in enumerate(episode_indices):
            if episode_id != current_episode_id:
                # Finalize previous episode
                if current_episode_id != -1:
                    self.episode_metadata[current_episode_id] = {
                        'start': start_frame,
                        'end': i,
                        'episode_index': current_episode_id
                    }

                # Start new episode
                current_episode_id = episode_id
                start_frame = i

        if current_episode_id != -1:
            self.episode_metadata[current_episode_id] = {
                'start': start_frame,
                'end': len(hf_ds),
                'episode_index': current_episode_id
            }

        print(f"LeRobot: Manual computation complete. Found {len(self.episode_metadata)} episodes.")


        if not self.episode_metadata:
            raise RuntimeError(
                f"\n--- LeRobot Dataset Metadata Error ---\n"
                f"Failed to compute episode metadata for repo_id: '{repo_id}'.\n"
                f"The resulting episode dictionary is empty. Please check the dataset integrity.\n"
                f"----------------------------------------"
            )

        episode_keys = sorted(list(self.episode_metadata.keys()))
        num_episodes = len(episode_keys)

        split_idx = int(num_episodes * train_fraction)

        if split == "train":
            self.traj_indices = episode_keys[:split_idx]
        elif split == "val":
            self.traj_indices = episode_keys[split_idx:]
        else:
            raise ValueError(f"Unknown split: {split}")

        print(f"LeRobot: Loaded {len(self.traj_indices)} episodes for split '{split}'.")

        # Determine the observation key
        # Using a common visual key for Aloha/Insertion
        self.obs_key = "observation.images.top" if visual_input else "observation.state"

        # Simple check for key existence (optional, but good for robustness)
        if len(self.full_dataset) > 0 and self.obs_key not in self.full_dataset[0]:
            print(f"Warning: Primary key '{self.obs_key}' not found. Check dataset keys.")

    def __len__(self) -> int:
        """Returns the number of trajectories (episodes) in this split."""
        return len(self.traj_indices)

    def get_seq_length(self, idx: int) -> int:
        """Returns the length of the idx-th trajectory in this split."""
        episode_key = self.traj_indices[idx]
        data = self.episode_metadata[episode_key]
        return data['end'] - data['start']

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns the full trajectory for the given index: (observations, actions, mask, [goals])
        """
        episode_key = self.traj_indices[idx]
        metadata = self.episode_metadata[episode_key]
        start, end = metadata['start'], metadata['end']
        T = end - start

        # 1. Fetch the slice of the LeRobot SampleDataset corresponding to the whole episode
        # IMPORTANT: Use the frame indices (start, end) on the full_dataset which is a torch Dataset
        episode_samples = [self.full_dataset[i] for i in range(start, end)]

        # Helper to stack and move to device
        def process_tensor(key):
            data_list = [sample[key] for sample in episode_samples]

            # Stack all frames along the time dimension (T)
            if isinstance(data_list[0], np.ndarray):
                tensor = torch.from_numpy(np.stack(data_list, axis=0)).float().to(self.device)
            elif torch.is_tensor(data_list[0]):
                tensor = torch.stack(data_list, dim=0).float().to(self.device)
            else:
                # Check if it's an integer array-like (e.g., episode_index)
                if all(isinstance(x, (int, np.integer)) for x in data_list):
                    tensor = torch.as_tensor(data_list).long().to(self.device)
                else:
                    raise TypeError(f"Unsupported data type for key {key}")

            return tensor

        # 2. Extract and format data
        observations = process_tensor(self.obs_key)
        actions = process_tensor('action')

        # 3. Create Mask
        mask = torch.ones(T, dtype=torch.bool).to(self.device)

        # 4. Handle Goals (Skipped if onehot_goals is False)
        tensors = [observations, actions, mask]

        if self.onehot_goals:
            # Need to fetch task_index; usually only present once per episode or constant
            # We assume it's constant across the episode, so we check the first sample
            if 'task_index' in episode_samples[0]:
                # Assuming task_index is consistent across the episode
                task_index = episode_samples[0]['task_index']

                # If task_index is a numpy array (e.g., [1]), extract the scalar
                if isinstance(task_index, np.ndarray):
                    task_index = task_index.item()

                onehot = F.one_hot(torch.as_tensor(task_index).long(), num_classes=self.GOAL_DIM).float()
                goals = einops.repeat(onehot, 'd -> t d', t=T).to(self.device)
                tensors.append(goals)
            else:
                raise NotImplementedError("Goal conditioning (onehot_goals=True) requires 'task_index'.")

        # Return the required tuple
        return tuple(tensors)

# --- SLICER CLASS ---

class TrajectorySlicerDataset(TrajectoryDataset):
    """
    Slices the full trajectory into windows (observation history + action chunk).
    This logic is directly ported from your previous turn.
    """

    def __init__(
            self,
            dataset: TrajectoryDataset,
            window: int,
            action_window: int,
            vqbet_get_future_action_chunk: bool = True,
            future_conditional: bool = False,
            get_goal_from_dataset: bool = False,
            min_future_sep: int = 0,
            future_seq_len: Optional[int] = None,
            only_sample_tail: bool = False,
            transform: Optional[Callable] = None,
    ):
        if future_conditional and future_seq_len is None:
            raise ValueError("must specify a future_seq_len when future_conditional is True")

        self.dataset = dataset
        self.window = window
        self.action_window = action_window
        self.vqbet_get_future_action_chunk = vqbet_get_future_action_chunk
        self.future_conditional = future_conditional
        self.get_goal_from_dataset = get_goal_from_dataset
        self.min_future_sep = min_future_sep
        self.future_seq_len = future_seq_len
        self.only_sample_tail = only_sample_tail
        self.transform = transform
        self.slices = []
        min_seq_length = np.inf

        if vqbet_get_future_action_chunk:
            min_window_required = window + action_window
        else:
            min_window_required = max(window, action_window)

        for i in range(len(self.dataset)):
            T = self.dataset.get_seq_length(i)
            min_seq_length = min(T, min_seq_length)

            if T - min_window_required < 0:
                print(
                    f"Ignored short sequence #{i}: len={T}, window={min_window_required}"
                )
            else:
                # Padding slices (slices shorter than window, padded by duplicating the start frame)
                # These are used for VQ-BET mode
                if vqbet_get_future_action_chunk:
                    self.slices += [
                        (i, 0, end + 1) for end in range(window - 1)
                    ]

                # Full slices
                self.slices += [
                    (i, start, start + window)
                    for start in range(T - min_window_required)
                ]

        if min_seq_length < min_window_required:
            print(
                f"Ignored short sequences. To include all, set window <= {min_seq_length}."
            )

    def get_seq_length(self, idx: int) -> int:
        if self.future_conditional:
            return self.future_seq_len + self.window
        else:
            return self.window

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]

        # Retrieve the full trajectory from the underlying dataset
        traj = self.dataset[i]

        # --- Slice Observation and Action (Past Window) ---
        if self.vqbet_get_future_action_chunk:
            # VQ-BET mode: pads observation window and gets the action chunk
            if end - start < self.window:
                # Handle padding for short sequences (at the start of the episode)
                pad_len = self.window - (end - start)

                # Observation padding
                obs_data = traj[0]
                obs_slice = obs_data[start:end]
                pad_obs_single = obs_data[start].unsqueeze(0)

                # Determine padding shape based on data dimension
                if obs_data.ndim > 2:  # Image/High-dim observation (T, C, H, W, etc.)
                    pad_obs = torch.tile(pad_obs_single, (pad_len,) + tuple(1 for _ in range(obs_data.ndim - 1)))
                else:  # State observation (T, Dim)
                    pad_obs = torch.tile(pad_obs_single, (pad_len, 1))

                values = [
                    torch.cat((pad_obs, obs_slice), dim=0),
                    # Action padding: pad_len actions + future actions
                    torch.cat((
                        torch.tile(traj[1][start].unsqueeze(0), (pad_len, 1)),  # Tile first action
                        traj[1][start: end - 1 + self.action_window]
                    ), dim=0)
                ]
            else:
                # Full window slice
                values = [
                    traj[0][start:end],
                    traj[1][start: end - 1 + self.action_window],
                ]
        else:
            # Standard sequential model mode (single observation + action chunk)
            values = [
                torch.unsqueeze(traj[0][start], dim=0),  # Current observation only
                traj[1][start: start + self.action_window],  # Future action chunk
            ]

        # --- Handle Goal/Future Conditional ---
        if self.get_goal_from_dataset or self.future_conditional:
            T_traj = self.dataset.get_seq_length(i)

            # Determine the goal/future observation index
            # If get_goal_from_dataset (one-hot), it's the 4th element (index 3)
            # If future_conditional (future obs), it's the 1st element (index 0, observations)
            goal_data_idx = 3 if self.get_goal_from_dataset else 0

            # future_obs_data is either goals (index 3) or observations (index 0)
            future_obs_data = traj[goal_data_idx]

            # --- Future Sampling Logic ---
            valid_start_range = (
                end + self.min_future_sep,
                T_traj - (self.future_seq_len if self.future_seq_len is not None else 0),
            )

            # If a valid range exists
            if valid_start_range[0] < valid_start_range[1]:
                if self.only_sample_tail:
                    future_obs = future_obs_data[-self.future_seq_len:]
                else:
                    low, high = valid_start_range
                    start_fut = np.random.randint(low, high)
                    end_fut = start_fut + self.future_seq_len
                    future_obs = future_obs_data[start_fut:end_fut]
            else:
                # Fallback: take the end of the episode or use zeros
                if self.get_goal_from_dataset:
                    # One-hot goal is constant across T, just take the end
                    future_obs = future_obs_data[-self.future_seq_len:]
                else:
                    # Future observation goal: take the end of the episode
                    future_obs = self.dataset[i][0][-self.future_seq_len:]

            values.append(future_obs)

        # optionally apply transform
        if self.transform is not None:
            values = self.transform(values)

        # Ensure the final tuple is correctly unpacked
        return tuple(values)


# --- SPLIT FUNCTIONS (For compatibility) ---

def random_split_traj(
        dataset: TrajectoryDataset,
        lengths: Sequence[int],
        generator: Optional[torch.Generator] = default_generator,
) -> List[TrajectorySubset]:
    """Randomly split a trajectory dataset into non-overlapping new datasets of given lengths."""
    if sum(lengths) != len(dataset):
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()
    return [
        TrajectorySubset(dataset, indices[offset - length: offset])
        for offset, length in zip(itertools.accumulate(lengths), lengths)  # FIXED: Use itertools.accumulate
    ]


# --- FINAL ENTRY POINT FUNCTION (The one Hydra targets) ---

def get_lerobot_train_val(
        repo_id: str,
        train_fraction: float = 0.9,
        window_size: int = 10,
        action_window_size: int = 10,
        vqbet_get_future_action_chunk: bool = True,
        only_sample_tail: bool = False,
        goal_conditional: Optional[str] = None,  # Default: No goal conditioning
        future_seq_len: Optional[int] = None,
        min_future_sep: int = 0,
        transform: Optional[Callable[[Any], Any]] = None,
        visual_input: bool = True,  # Default to True for insertion task
        device: str = "cuda",
        goal_dim: int = 5,
        **kwargs
):
    """
    Initializes the LeRobotDataset, splits the trajectories by episode,
    and wraps them in the TrajectorySlicerDataset to create train/val splits.

    This function will be targeted by Hydra (e.g., _target_: your_module.get_relay_kitchen_train_val).
    """
    # Disable future-based goal logic if goal_conditional is None
    is_future_conditional = (goal_conditional == "future")
    is_onehot_goal = (goal_conditional == "onehot")

    # Assert checks only if goal conditioning is enabled
    if goal_conditional is not None:
        assert goal_conditional in ["future", "onehot"], "goal_conditional must be 'future', 'onehot', or None"
        if is_future_conditional and future_seq_len is None:
            raise ValueError("future_seq_len must be set for 'future' goal_conditional.")

    # 1. Instantiate the base dataset for train split
    full_dataset_train = InsertionAlohaTrajectoryDataset(
        repo_id=repo_id,
        split="train",
        train_fraction=train_fraction,
        device=device,
        onehot_goals=is_onehot_goal,
        visual_input=visual_input,
        goal_dim=goal_dim,
        **kwargs
    )
    # 2. Instantiate the base dataset for val split
    full_dataset_val = InsertionAlohaTrajectoryDataset(
        repo_id=repo_id,
        split="val",
        train_fraction=train_fraction,
        device=device,
        onehot_goals=is_onehot_goal,
        visual_input=visual_input,
        goal_dim=goal_dim,
        **kwargs
    )

    # 3. Slice the datasets
    traj_slicer_kwargs = {
        "window": window_size,
        "action_window": action_window_size,
        "vqbet_get_future_action_chunk": vqbet_get_future_action_chunk,
        "future_conditional": is_future_conditional,
        "get_goal_from_dataset": is_onehot_goal,
        "min_future_sep": min_future_sep,
        "future_seq_len": future_seq_len,
        "only_sample_tail": only_sample_tail,
        "transform": transform,
    }

    train_slices = TrajectorySlicerDataset(full_dataset_train, **traj_slicer_kwargs)
    val_slices = TrajectorySlicerDataset(full_dataset_val, **traj_slicer_kwargs)

    return train_slices, val_slices


if __name__ == '__main__':
    # --- Example Usage for Aloha Insertion (No Goal Conditioning) ---
    # NOTE: You must have the dataset cloned locally or ensure your machine
    # has Hugging Face credentials to download it.
    REPO_ID = "lerobot/aloha_sim_insertion_human"

    print("\n--- Testing LeRobot Aloha Dataset Loading (No Goal Conditioning) ---")

    # We use CPU for testing, change to 'cuda' for actual training
    # Setting visual_input=True (default for insertion)

    train_ds, val_ds = get_lerobot_train_val(
        repo_id=REPO_ID,
        window_size=10,
        action_window_size=10,
        vqbet_get_future_action_chunk=True,
        visual_input=True,
        device="cpu"
    )
    print(f"\nTraining Dataset (Sliced): {len(train_ds)} samples")
    print(f"Validation Dataset (Sliced): {len(val_ds)} samples")

    # Access a sample (should return (observations, actions, mask) - 3 elements)
    sample = train_ds[0]
    print(f"\nSample 0 (Windowed Trajectory) has {len(sample)} elements.")

    obs, action, mask = sample[0], sample[1], sample[2]
    print(f"Observation window shape (T, C, H, W): {obs.shape}")
    print(f"Action chunk shape (T_act, Dim): {action.shape}")
    print(f"Mask shape (T_obs): {mask.shape}")

