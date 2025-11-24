import numpy as np
import math
from gym import utils
from gym.envs.mujoco import mujoco_env
import gym
#TODO

class AlohaInsertionEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    # This should point to your specific ALOHA model file
    FILE = "aloha.xml"

    # Define the default frame skip (e.g., 5 steps per action)
    DEFAULT_CAMERA_NAME = "main_cam"

    # The number of steps to skip per action.
    FRAME_SKIP = 5

    def __init__(self, file_path=None, seed=0):
        # Initialize parent classes
        mujoco_env.MujocoEnv.__init__(self, file_path or self.FILE, self.FRAME_SKIP)
        utils.EzPickle.__init__(self)

        self.rng = np.random.RandomState(seed)

        self.env = gym.make("gym_aloha/AlohaInsertion-v0", obs_type="pixels_agent_pos")
        self.obs = None
        self.reward = 0
        self.terminated = False
        print(f"Aloha Insertion Environment initialized.")
        print(f"Action space dimension (nv): {self.model.nv}")

    # --- Core Environment Methods ---

    def step(self, action):
        """
        Applies the action and advances the simulation.
        """
        obs, reward, terminated, truncated, info = self.env.step(action_np)
        self.obs = obs
        self.reward = reward
        self.terminated = terminated
        return obs, reward, terminated, info

    def reset_model(self):
        """
        Resets the robot to a starting configuration with slight randomization.
        """
        obs, info = self.env.reset()
        self.obs = obs
        return obs

    def _get_obs(self):
        """
        Gathers the necessary state and vision observations.
        For VQ-BeT, the primary observation is the image, but we include state here.
        """
        return self.obs

    # --- Reward and Termination Logic ---

    def _get_reward_and_info(self):
        """
        Calculates the reward for the current timestep based on task progress.
        """

        info_rewards = {
            "reward" : 0
        }

        return self.reward, info_rewards

    def _is_done(self):
        """
        Checks if the episode should terminate.
        """
        return self.terminated

    # def get_body_com(self, name):
    #     """Get the center of mass (COM) of a specific body."""
    #     body_id = self.model.body_names.index(name)
    #     return self.data.body_xpos[body_id]

    # ... You can add other helper methods like rendering setup, etc.

# NOTE: The actual VQ-BeT training pipeline will use a DataLoader to wrap this
# environment or load the pre-recorded LeRobotDataset, as VQ-BeT is a
# Offline Reinforcement Learning / Imitation Learning algorithm.