"""Environments using ant robot."""
import logging
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import gymnasium as gym
from gymnasium.envs.registration import register


def get_split_idx(l, seed, train_fraction=0.95):
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(l, generator=rng).tolist()
    l_train = int(l * train_fraction)
    return idx[:l_train], idx[l_train:]


class ALOHAWRAPPER(gym.Wrapper):
    def __init__(self, env, id, goal_cond=False):
        super(AlohaWrapper, self).__init__(env)
        self.id = id
        self.env = env
        # if goal_cond:
        #     self.env.set_goalcond()

    def reset(self, *args, **kwargs):
        obs = self.env.reset()
        self.num_achieved = 0
        # if self.env.goal_cond:
        #     one_indices = np.random.choice(4, 2, replace=False)
        #     self.env.set_achieved(one_indices)
        #     self.num_achieved = 2
        print("Aloha insertion episode start!")
        #Todo check
        return_obs = np.concatenate((obs["observation"], obs["for_vq_bet"]))
        return_obs[29:37] = 0
        return return_obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        return_obs = np.concatenate((obs["observation"], obs["for_vq_bet"]))
        return_obs[29:37] = 0
        return return_obs, reward, done, info


register(
    id="AlohaInsertion-eval-v0",
    entry_point="envs.Aloha.aloha_insertion:AlohaInsertionEnv",
    max_episode_steps=1200,
    reward_threshold=0.0,
)
