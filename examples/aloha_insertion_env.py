import logging
from typing import Callable, Dict, Optional, Tuple, Union
import numpy as np
import torch
import gymnasium as gym
import gym_aloha
from gymnasium.envs.registration import register
from dataset import InsertionAlohaTrajectoryDataset
import einops

# Setup logging if it hasn't been configured elsewhere
logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def get_split_idx(l, seed, train_fraction=0.95):
    """Utility function for splitting dataset indices."""
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(l, generator=rng).tolist()
    l_train = int(l * train_fraction)
    return idx[:l_train], idx[l_train:]


class ALOHAWRAPPER(gym.Wrapper):
    """
    Custom wrapper for the Aloha Insertion environment to modify the observation
    space and handle data structure necessary for VQ-BET.
    """

    def __init__(self, id, goal_cond=False, **kwargs):
        self.id = id
        env = gym.make("gym_aloha/AlohaInsertion-v0")
        self.env = env

        # --- Goal-Conditioning Attributes ---
        self.goal_masking = True
        self._goal_onehot = None
        self.tasks_to_complete = ["insertion"]
        # ------------------------------------

        super().__init__(env)

    # --- Goal-Conditioning Methods (copied and adapted from Kitchen env) ---
    def set_task_goal(self, one_hot_indices):
        """
        Sets the goal state for the environment using the one-hot goal from the dataset.
        For ALOHA Insertion, this primarily records the goal vector.
        """
        self._goal_onehot = one_hot_indices
        # logging.info("ALOHA set_task_goal called (single-task). Goal onehot: {}".format(one_hot_indices))

    def set_goal_masking(self, goal_masking=True):
        """Sets goal masking for goal-conditioned approaches (like RPL)."""
        self.goal_masking = goal_masking

    # -----------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        print(obs)
        obs = obs.cpu().numpy()
        self.num_achieved = 0
        print("Aloha insertion episode start!")

        # The observation structure is expected to be (image, low-dim-state)
#        image = self.env.render()
#        return_obs = (image, obs)

        # Return the modified observation and the info dict
        return obs, info

    def step(self, action):
        # The Gymnasium API returns (observation, reward, terminated, truncated, info)
        observation, reward, terminated, truncated, info = self.env.step(action)
        print(obs)
        obs = obs.cpu().numpy()
        #image = self.env.render()
        #return_obs = (image, observation)
        done = terminated or truncated

        return obs, reward, done, info


# Registration for use with hydra/gym.make
register(
    id="AlohaInsertion-eval-v0",
    entry_point="aloha_insertion_env:ALOHAWRAPPER",
    max_episode_steps=1200,
    reward_threshold=0.0,
)


def get_goal_fn(
        data_directory: str,
        goal_conditional: Optional[str] = None,
        goal_seq_len: Optional[int] = None,
        seed: Optional[int] = None,
        train_fraction: Optional[float] = None,
        unconditional: bool = False,
        goal_dim=60,
        visual_input=False,
) -> Callable[
    [gym.Env, torch.Tensor, torch.Tensor, torch.Tensor],
    Union[
        torch.Tensor,
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
        None,
    ],
]:
    empty_tensor = torch.zeros(0)
    if unconditional:
        return lambda env, state, goal_idx, frame_idx: (empty_tensor, {})

    # Assuming 'split="train"' is necessary based on your partial code
    relay_traj = InsertionAlohaTrajectoryDataset(
        data_directory, split="train", onehot_goals=True, visual_input=visual_input
    )

    train_idx, val_idx = get_split_idx(
        len(relay_traj),
        seed=seed,
        train_fraction=train_fraction or 1.0,
    )
    goal_fn = lambda env, state, goal_idx, frame_idx: None

    if goal_conditional == "future":
        assert (
                goal_seq_len is not None
        ), "goal_seq_len must be provided if goal_conditional is 'future'"

        def future_goal_fn(env, state, goal_idx, frame_idx):  # type: ignore
            # Assuming the dataset returns at least (obs, action, reward, onehot, ...)
            # We use obs, onehot, and ignore the other values with underscores/rest.
            obs, _, _, onehot, *rest = relay_traj[train_idx[goal_idx]]  # seq_len x obs_dim, seq_len x onehot_dim
            info = {}
            if frame_idx == 0:
                # Use einops for max-pooling across the time dimension 'T'
                onehot = einops.reduce(onehot, "T C -> C", "max")
                info["onehot_goal"] = onehot

                goal_cond = goal_dim > 0
                if not goal_cond:
                    # Assuming 7 is the onehot goal dimension, which is typical for the kitchen environment
                    # but should be adjusted if ALOHA uses a different onehot size (e.g., 1 for single task)
                    onehot = torch.ones(7).cuda()

                    # env.set_task_goal is now available on the ALOHAWRAPPER
                env.set_task_goal(onehot)

            obs = obs[-goal_seq_len:]
            return obs, info

        goal_fn = future_goal_fn

    elif goal_conditional == "onehot":

        def onehot_goal_fn(env, state, goal_idx, frame_idx):
            if frame_idx == 0:
                logging.info(f"goal_idx: {train_idx[goal_idx]}")

            # Assuming the dataset returns at least (obs, action, reward, onehot, ...)
            _, _, _, onehot_goals, *rest = relay_traj[train_idx[goal_idx]]  # seq_len x onehot_dim

            # Use the onehot goal at the current frame index (or the last frame if sequence ends)
            return onehot_goals[min(frame_idx, len(onehot_goals) - 1)], {}

        goal_fn = onehot_goal_fn

    else:
        raise ValueError(f"goal_conditional {goal_conditional} not recognized")

    return goal_fn


if __name__ == "__main__":
    env = ALOHAWRAPPER(id="TestEnv")
    processed_obs, info = env.reset()
    print(processed_obs, info)
    print("Successfully reset environment.")

    dummy_action = np.zeros(14)
    obs, reward, done, info = env.step(dummy_action)
    print(obs)
    print(f"Step reward: {reward}")
