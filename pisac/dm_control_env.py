# coding=utf-8
# Copyright 2020 The PI-SAC Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DM control environment wrappers."""

import collections
import copy

from dm_control import suite
from dm_control.suite.wrappers import pixels as pixel_wrapper

import gin
import numpy as np

from tf_agents.environments import dm_control_wrapper
from tf_agents.environments import wrappers
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


STATE_KEY = 'state'


@gin.configurable
def load(domain_name,
         task_name,
         task_kwargs=None,
         environment_kwargs=None,
         env_load_fn=suite.load,  # use custom_suite.load for customized env
         action_repeat_wrapper=wrappers.ActionRepeat,
         action_repeat=1,
         frame_stack=4,
         episode_length=1000,
         actions_in_obs=True,
         rewards_in_obs=False,
         pixels_obs=True,
         # Render params
         grayscale=False,
         visualize_reward=False,
         render_kwargs=None):
  """Returns an environment from a domain name, task name."""
  env = env_load_fn(domain_name, task_name, task_kwargs=task_kwargs,
                    environment_kwargs=environment_kwargs,
                    visualize_reward=visualize_reward)
  if pixels_obs:
    env = pixel_wrapper.Wrapper(
        env, pixels_only=False, render_kwargs=render_kwargs)

  env = dm_control_wrapper.DmControlWrapper(env, render_kwargs)

  if pixels_obs and grayscale:
    env = GrayscaleWrapper(env)
  if action_repeat > 1:
    env = action_repeat_wrapper(env, action_repeat)
  if pixels_obs:
    env = FrameStack(env, frame_stack, actions_in_obs, rewards_in_obs)
  else:
    env = FlattenState(env)

  # Adjust episode length based on action_repeat
  max_episode_steps = (episode_length + action_repeat - 1) // action_repeat

  # Apply a time limit wrapper at the end to properly trigger all reset()
  env = wrappers.TimeLimit(env, max_episode_steps)
  return env


@gin.configurable
class GrayscaleWrapper(wrappers.PyEnvironmentBaseWrapper):
  """Convert RGB observations to grayscale."""

  def __init__(self, env):
    """Initializes a grayscale wrapper."""
    super(GrayscaleWrapper, self).__init__(env)

    # Update the observation spec in the environment.
    observation_spec = env.observation_spec()

    # Update the observation spec.
    self._grayscale_observation_spec = copy.copy(observation_spec)
    frame_shape = observation_spec['pixels'].shape
    grayscale_frame_shape = frame_shape[:2] + (1,)
    self._grayscale_observation_spec['pixels'] = array_spec.ArraySpec(
        shape=grayscale_frame_shape,
        dtype=observation_spec['pixels'].dtype,
        name='grayscale_pixels')

  def _grayscale_observation_timestep(self, timestep):
    """Convert observations to grayscale."""
    observations = timestep.observation
    im = observations['pixels']
    im_gray = im[:, :, 0] * 0.299 + im[:, :, 1] * 0.587 + im[:, :, 2] * 0.114
    im_gray = np.array(im_gray, dtype=np.uint8)
    observations['pixels'] = np.expand_dims(im_gray, 2)
    return ts.TimeStep(
        timestep.step_type, timestep.reward, timestep.discount,
        observations)

  def _step(self, action):
    """Steps the environment while converting observations to grayscale.

    Args:
      action: A NumPy array, or a nested dict, list or tuple of arrays
        corresponding to `action_spec()`.

    Returns:
      A `TimeStep` namedtuple containing:
        step_type: A `StepType` value.
        reward: Reward at this timestep.
        discount: A discount in the range [0, 1].
        observation: A flattened NumPy array of shape corresponding to
         `observation_spec()`.
    """
    return self._grayscale_observation_timestep(self._env.step(action))

  def _reset(self):
    """Starts a new sequence and returns the first `TimeStep` of this sequence.

    Returns:
      A `TimeStep` namedtuple containing:
        step_type: A `StepType` of `FIRST`.
        reward: `None`, indicating the reward is undefined.
        discount: `None`, indicating the discount is undefined.
        observation: A flattened NumPy array of shape corresponding to
         `observation_spec()`.
    """
    return self._grayscale_observation_timestep(self._env.reset())

  def observation_spec(self):
    """Defines the observations provided by the environment.

    Returns:
      An `ArraySpec` with a shape of the total length of observations kept.
    """
    return self._grayscale_observation_spec


@gin.configurable
class FrameStack(wrappers.PyEnvironmentBaseWrapper):
  """Stack frames."""

  def __init__(self, env, stack_size, actions_in_obs, rewards_in_obs):
    """Initializes a wrapper."""
    super(FrameStack, self).__init__(env)
    self.stack_size = stack_size
    self._frames = collections.deque(maxlen=stack_size)
    self.actions_in_obs = actions_in_obs
    self.rewards_in_obs = rewards_in_obs

    # Update the observation spec in the environment.
    observation_spec = env.observation_spec()

    # Update the observation spec.
    self._new_observation_spec = copy.copy(observation_spec)

    # Redefine pixels spec
    frame_shape = observation_spec['pixels'].shape
    stacked_frame_shape = frame_shape[:2] + (frame_shape[2]*stack_size,)
    self._new_observation_spec['pixels'] = array_spec.ArraySpec(
        shape=stacked_frame_shape,
        dtype=observation_spec['pixels'].dtype,
        name='grayscale_pixels')

    # Define action stack spec
    if self.actions_in_obs:
      self._actions = collections.deque(maxlen=stack_size-1)
      stacked_action_shape = (stack_size - 1,) + env.action_spec().shape
      self._new_observation_spec['actions'] = array_spec.ArraySpec(
          shape=stacked_action_shape,
          dtype=env.action_spec().dtype,
          name='actions')

    # Define rewards stack spec
    if self.rewards_in_obs:
      self._rewards = collections.deque(maxlen=stack_size)
      self._new_observation_spec['rewards'] = array_spec.ArraySpec(
          shape=(stack_size,),
          dtype=np.float32,
          name='rewards')

  def _step(self, action):
    """Steps the environment."""
    time_step = self._env.step(action)
    observations = time_step.observation

    # frame stacking
    self._frames.append(observations['pixels'])
    observations['pixels'] = np.concatenate(self._frames, axis=2)

    # action stacking
    if self.actions_in_obs:
      self._actions.append(action)
      observations['actions'] = np.stack(self._actions)

    # reward stacking
    if self.rewards_in_obs:
      self._rewards.append(time_step.reward)
      observations['rewards'] = np.stack(self._rewards)

    return ts.TimeStep(
        time_step.step_type, time_step.reward, time_step.discount,
        observations)

  def _reset(self):
    """Starts a new sequence and returns the first `TimeStep`."""
    time_step = self._env.reset()
    observations = time_step.observation

    # initial frame stacking
    for _ in range(self.stack_size):
      self._frames.append(observations['pixels'])
    observations['pixels'] = np.concatenate(self._frames, axis=2)

    # initial action stacking
    if self.actions_in_obs:
      for _ in range(self.stack_size-1):
        self._actions.append(np.zeros(self._env.action_spec().shape,
                                      dtype=np.float32))
      observations['actions'] = np.stack(self._actions)

    # initial reward stacking
    if self.rewards_in_obs:
      for _ in range(self.stack_size):
        self._rewards.append(np.array(0.0, dtype=np.float32))
      observations['rewards'] = np.stack(self._rewards)

    return ts.TimeStep(
        time_step.step_type, time_step.reward, time_step.discount,
        observations)

  def observation_spec(self):
    """Defines the observations provided by the environment."""
    return self._new_observation_spec


@gin.configurable
class FlattenState(wrappers.PyEnvironmentBaseWrapper):
  """Stack frames and drop other observations."""

  def __init__(self, env):
    """Initializes a wrapper."""
    super(FlattenState, self).__init__(env)
    # Update the observation spec in the environment.
    observation_spec = env.observation_spec()

    dim = 0
    dtype = None
    for v in observation_spec.values():
      dim += v.shape[0]
      dtype = v.dtype

    self._new_observation_spec = array_spec.ArraySpec(
        shape=(dim,),
        dtype=dtype,
        name='state')

  def _flatten_obs(self, obs):
    """Flatten and concatentate states in the observation dict."""
    obs_list = []
    for v in obs.values():
      flat = np.array([v]) if np.isscalar(v) else v.ravel()
      obs_list.append(flat)
    return np.concatenate(obs_list, axis=-1)

  def _step(self, action):
    """Steps the environment."""
    time_step = self._env.step(action)
    concat_obs = self._flatten_obs(time_step.observation)
    return ts.TimeStep(
        time_step.step_type, time_step.reward, time_step.discount, concat_obs)

  def _reset(self):
    """Starts a new sequence and returns the first `TimeStep`."""
    time_step = self._env.reset()
    concat_obs = self._flatten_obs(time_step.observation)
    return ts.TimeStep(
        time_step.step_type, time_step.reward, time_step.discount, concat_obs)

  def observation_spec(self):
    """Defines the observations provided by the environment."""
    return self._new_observation_spec
