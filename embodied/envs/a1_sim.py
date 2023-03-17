"""Wrapper to make the a1 environment suitable for OpenAI gym."""
import gym

# from motion_imitation.envs import env_builder
from motion_imitation.robots import a1
from motion_imitation.robots import robot_config

import embodied
import numpy as np

from . import gym


class A1Sim(embodied.Env):

  def __init__(self, task, repeat=1, length=1000, resets=True):
    assert task in ('sim', 'real'), task
    import motion_imitation.envs.env_builder as env_builder
    self._env = env_builder.build_regular_env(
        a1.A1,
        motor_control_mode=robot_config.MotorControlMode.POSITION,
        enable_rendering=False,
        action_limit=True,
        on_rack=False)
    self.length = length
    self.sim_step = 0

  @property
  def obs_space(self):
    space = self._env.observation_space
    return {
        'vector': embodied.Space(np.float32, shape=space.shape, low=space.low, high=space.high)
    }

  @property
  def act_space(self):
    # return self._env.act_space
    return {
        'action': embodied.Space(np.float32, (12,), -1.0, 1.0),
        'reset': embodied.Space(bool, ()),
    }

  def step(self, action):
    # if action["reset"]: # TODO: ?
    #   self._env.reset()
    self.sim_step += 1
    obs = self._env.step(action["action"])
    if self.sim_step >= self.length:
      self._env.reset()
      self.sim_step = 0
    return {
      "vector": obs[0], 
      'reward': obs[1],
      'is_first': obs[3]['is_first'], # action["reset"], # obs[3]["is_first"], 
      'is_last': self.sim_step==0, # action["reset"], # obs[2],
      'image': self._env.render(mode='rgb_array')
    }
