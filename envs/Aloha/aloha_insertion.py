
import math
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
import gymnasium as gym

def q_inv(a):
    return [a[0], -a[1], -a[2], -a[3]]


def q_mult(a, b):  # multiply two quaternion
    w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    i = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    j = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    k = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    return [w, i, j, k]


class AlohaInsertionEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    FILE = "Aloha.xml"
    ORI_IND = 6

    def __init__(
        self,
    ):

        mujoco_env.MujocoEnv.__init__(self, file_path, 5)
        utils.EzPickle.__init__(self)

    @property
    def physics(self):
        # check mujoco version is greater than version 1.50 to call correct physics
        # model containing PyMjData object for getting and setting position/velocity
        # check https://github.com/openai/mujoco-py/issues/80 for updates to api
        if mujoco_py.get_version() >= "1.50":
            return self.sim
        else:
            return self.model

    def _step(self, a):
        return self.step(a)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)

        # stochasticity
        # qpos = np.copy(self.physics.data.qpos)
        # qvel = np.copy(self.physics.data.qvel)
        # qpos[:2] += self.rng.normal(scale=0.05, size=2)
        # self.set_state(qpos, qvel)

        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.5 * np.square(a).sum()
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_survive=survive_reward,
            ),
        )

    def _get_obs(self):
        # No cfrc observation
        if self._expose_all_qpos:
            obs = np.concatenate(
                [
                    self.data.qpos.flat[:15],  # Ensures only ant obs.
                    self.data.qvel.flat[:14],
                ]
            )
        else:
            obs = np.concatenate(
                [
                    self.data.qpos.flat[2:15],
                    self.data.qvel.flat[:14],
                ]
            )

        if self._expose_body_coms is not None:
            for name in self._expose_body_coms:
                com = self.get_body_com(name)
                if name not in self._body_com_indices:
                    indices = range(len(obs), len(obs) + len(com))
                    self._body_com_indices[name] = indices
                obs = np.concatenate([obs, com])

        if self._expose_body_comvels is not None:
            for name in self._expose_body_comvels:
                comvel = self.get_body_comvel(name)
                if name not in self._body_comvel_indices:
                    indices = range(len(obs), len(obs) + len(comvel))
                    self._body_comvel_indices[name] = indices
                obs = np.concatenate([obs, comvel])
        return obs

    def reset_model(self):
        qpos = self.init_qpos + self.rng.uniform(size=self.model.nq, low=-0.1, high=0.1)
        qvel = self.init_qvel + self.rng.randn(self.model.nv) * 0.1

        # Set everything other than ant to original position and 0 velocity.
        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def get_ori(self):
        ori = [0, 1, 0, 0]
        rot = self.physics.data.qpos[
            self.__class__.ORI_IND : self.__class__.ORI_IND + 4
        ]  # take the quaternion
        ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]  # project onto x-y plane
        ori = math.atan2(ori[1], ori[0])
        return ori

    def set_xy(self, xy):
        qpos = np.copy(self.physics.data.qpos)
        qpos[0] = xy[0]
        qpos[1] = xy[1]

        qvel = self.physics.data.qvel
        self.set_state(qpos, qvel)

    def get_xy(self):
        return self.physics.data.qpos[:2]

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 50
        self.viewer.cam.elevation = -90
