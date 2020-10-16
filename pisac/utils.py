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
"""Utils."""

import io

import gin
import matplotlib
matplotlib.use('Agg')
# pylint:disable=g-import-not-at-top
import matplotlib.pyplot as plt
import numpy as np
from qj_global import qj

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import composite
from tf_agents.utils import nest_utils

tfd = tfp.distributions


def filter_invalid_transition(trajectories, unused_arg1):
  """Filter invalid boundary transitions."""
  return ~tf.reduce_any(trajectories.is_boundary()[:-1])


def split_xy(traj, y_len, rewards_n_actions_only=False):
  """Split X (past) and Y (future) from a trjactory."""

  # Assume frame_stack = y_len = 4
  # len(traj) = frame_stack + 2
  # 0 0 0 0 1 2 3 4 5
  #|   x0  |
  #  |  x1   |
  #        |   y0  |
  #          |  y1   |
  #      | ra_y0 |
  #        | ra_y1 |

  r = traj.reward
  a = traj.action

  r_y0 = tf.nest.map_structure(
      lambda t: composite.slice_to(t, axis=1, end=y_len), r)
  r_y1 = tf.nest.map_structure(
      lambda t: composite.slice_from(t, axis=1, start=1), r)
  r_y1 = tf.nest.map_structure(
      lambda t: composite.slice_to(t, axis=1, end=y_len), r_y1)
  r_y0 = r_y0[:, None, ...]
  r_y1 = r_y1[:, None, ...]

  a_y0 = tf.nest.map_structure(
      lambda t: composite.slice_to(t, axis=1, end=y_len), a)
  a_y1 = tf.nest.map_structure(
      lambda t: composite.slice_from(t, axis=1, start=1), a)
  a_y1 = tf.nest.map_structure(
      lambda t: composite.slice_to(t, axis=1, end=y_len), a_y1)
  a_y0 = a_y0[:, None, ...]
  a_y1 = a_y1[:, None, ...]

  if rewards_n_actions_only:
    return r_y0, r_y1, a_y0, a_y1

  o = traj.observation

  o_x0 = tf.nest.map_structure(
      lambda t: composite.slice_to(t, axis=1, end=1), o)
  o_x1 = tf.nest.map_structure(
      lambda t: composite.slice_from(t, axis=1, start=1), o)
  o_x1 = tf.nest.map_structure(
      lambda t: composite.slice_to(t, axis=1, end=1), o_x1)

  o_y0 = tf.nest.map_structure(
      lambda t: composite.slice_from(t, axis=1, start=-2), o)
  o_y0 = tf.nest.map_structure(
      lambda t: composite.slice_to(t, axis=1, end=1), o_y0)
  o_y1 = tf.nest.map_structure(
      lambda t: composite.slice_from(t, axis=1, start=-1), o)
  return o_x0, o_x1, o_y0, o_y1, r_y0, r_y1, a_y0, a_y1


@gin.configurable
class MLP(network.Network):
  """2-layer fully connected network."""

  def __init__(self,
               input_tensor_spec,
               hidden_dims=(32, 5),
               activation=tf.keras.activations.relu,
               batch_squash=True,
               dtype=tf.float32,
               name='MLP'):
    """Creates an instance of `FCNet`."""
    super(MLP, self).__init__(
        input_tensor_spec=input_tensor_spec, state_spec=(), name=name)
    kernel_initializer = tf.compat.v1.variance_scaling_initializer(
        scale=2.0, mode='fan_in', distribution='truncated_normal')
    self._batch_squash = batch_squash
    self.fc = [
        tf.keras.layers.Dense(  # pylint: disable=g-complex-comprehension
            h,
            activation=activation if i < len(hidden_dims) - 2 else None,
            kernel_initializer=kernel_initializer,
            dtype=dtype,
            name='%s/dense%d' % (name, i))
        for i, h in enumerate(hidden_dims)]

  def call(self, obs, step_type=None, network_state=(), training=False):
    del step_type  # unused.
    if self._batch_squash:
      outer_rank = nest_utils.get_outer_rank(obs, self.input_tensor_spec)
      batch_squash = utils.BatchSquash(outer_rank)
      obs = tf.nest.map_structure(batch_squash.flatten, obs)
    states = obs
    for i in range(len(self.fc)):
      states = self.fc[i](states, training=training)
    if self._batch_squash:
      states = tf.nest.map_structure(batch_squash.unflatten, states)
    return states, network_state


@gin.configurable
class SimpleDeconv(network.Network):
  """Deconv network.."""

  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               filters=(128, 64, 32, 9),
               kernels=(3, 3, 11, 3),
               strides=(2, 2, 1, 2),
               paddings=('same', 'same', 'valid', 'same'),
               final_filters=None,
               activation=tf.keras.activations.relu,
               batch_squash=True,
               dtype=tf.float32,
               name='SimpleDeconv'):
    """Creates an instance of `SimpleDeconv`."""
    super(SimpleDeconv, self).__init__(
        input_tensor_spec=input_tensor_spec, state_spec=(), name=name)
    self._batch_squash = batch_squash

    if isinstance(paddings, str):
      paddings = (paddings,) * len(filters)
    if final_filters is None:
      filters = tuple(filters[:-1]) + (output_tensor_spec.shape[-1],)
    else:
      filters = tuple(filters[:-1]) + (final_filters,)

    proj_dim = output_tensor_spec.shape[0] ** (1.0 / 2 ** (len(filters) - 1))
    proj_dim = int(proj_dim) if proj_dim == int(proj_dim) else int(proj_dim) + 1
    proj_dim = max(int(np.ceil(np.sqrt(input_tensor_spec.shape[0]))), proj_dim)
    self.proj = tf.keras.layers.Dense(
        proj_dim ** 2,
        activation=None,
        dtype=dtype,
        name='%s/projection' % name)
    self.reshape = tf.keras.layers.Reshape([proj_dim, proj_dim, 1])
    self.deconv = [
        tf.keras.layers.Conv2DTranspose(  # pylint: disable=g-complex-comprehension
            filters=filters[i],
            kernel_size=kernels[i],
            strides=strides[i],
            padding=paddings[i],
            activation=activation if i < len(filters) - 2 else None,
            dtype=dtype,
            name='%s/deconv%d' % (name, i))
        for i in range(len(filters))]

  def call(self, obs, step_type=None, network_state=(), training=False):
    del step_type  # unused.
    if self._batch_squash:
      outer_rank = nest_utils.get_outer_rank(obs, self.input_tensor_spec)
      batch_squash = utils.BatchSquash(outer_rank)
      obs = tf.nest.map_structure(batch_squash.flatten, obs)
    states = obs
    states = self.proj(states)
    states = self.reshape(states)
    for i in range(len(self.deconv)):
      states = self.deconv[i](states, training=training)
    states = tf.nn.sigmoid(states)
    if self._batch_squash:
      states = tf.nest.map_structure(batch_squash.unflatten, states)
    return states, network_state


@gin.configurable
class PixelDecoder(tf.Module):
  """Decode pixels from states."""

  def __init__(self,
               deconv_net,
               optimizer,
               grad_clip=None,
               step_counter=None,
               observation_spec=None,
               image_summary_interval=10000,
               frame_stack=3,
               model_name='PixelDecoder'):
    self.deconv_net = deconv_net
    self.optimizer = optimizer
    self.grad_clip = grad_clip
    self.step_counter = step_counter
    self.observation_spec = observation_spec
    self.image_summary_interval = image_summary_interval
    self.frame_stack = frame_stack
    self.model_name = model_name

  def train(self, obs, z, prefiltered=True):
    variables_to_train = self.deconv_net.variables
    z = tf.stop_gradient(z)

    with tf.GradientTape() as tape:
      pred, _ = self.deconv_net(z, training=True)
      loss = tf.compat.v1.losses.mean_squared_error(obs, pred)
    grads = tape.gradient(loss, variables_to_train)
    grads_and_vars = tuple(zip(grads, variables_to_train))
    self.optimizer.apply_gradients(grads_and_vars)

    with tf.name_scope(self.model_name):
      tf.compat.v2.summary.scalar(
          name='loss',
          data=loss,
          step=self.step_counter)

      replay_summary(
          'y0', self.step_counter, reshape=True,
          frame_stack=self.frame_stack,
          image_summary_interval=self.image_summary_interval)(
              obs[:, None], None)
      replay_summary(
          'recon_y0', self.step_counter, reshape=True,
          frame_stack=self.frame_stack,
          image_summary_interval=self.image_summary_interval)(
              pred[:, None], None)
      replay_summary(
          'recon_y0_diff', self.step_counter, reshape=True,
          frame_stack=self.frame_stack,
          image_summary_interval=self.image_summary_interval)(
              ((obs - pred) / 2.0 + 0.5)[:, None], None)
    return loss


class Summ(object):
  """Directly save non tf.Tensor summaries."""

  def __init__(self, step, save_dir):
    self.step = step
    with tf.compat.v1.Graph().as_default():
      self.writer = tf.compat.v1.summary.FileWriterCache.get(save_dir)

  def scalar(self, tag, value):
    """Save a scalar summary."""
    summary = tf.compat.v1.Summary(
        value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
    self.writer.add_summary(summary, self.step)

  def image(self, tag, img):
    """Save an image summary."""
    if len(img.shape) < 3:
      img = img[..., None]
    if img.shape[-1] == 1:
      img = np.concatenate((img, img, img), axis=-1)

    s = io.StringIO()
    plt.imsave(s, img, format='png')

    img_sum = tf.compat.v1.Summary.Image(encoded_image_string=s.getvalue(),
                                         height=img.shape[0],
                                         width=img.shape[1])
    img_sum = tf.compat.v1.Summary.Value(tag=tag, image=img_sum)

    summary = tf.compat.v1.Summary(value=[img_sum])
    self.writer.add_summary(summary, self.step)

  def hist(self, tag, values, bins=200):
    """Save a histogram summary."""
    values = np.array(values, dtype=np.float32)
    if len(values) == 0:  # pylint: disable=g-explicit-length-test
      return

    try:
      values = np.nan_to_num(values)
      counts, edges = np.histogram(values, bins=bins)
      edges = edges[1:]

      hist = tf.compat.v1.HistogramProto()

      for edge in edges:
        hist.bucket_limit.append(edge)

      for c in counts:
        hist.bucket.append(c)

      hist.num = np.prod(values.shape)
      hist.min = values.min()
      hist.max = values.max()
      hist.sum = values.sum()
      hist.sum_squares = np.sum(values ** 2)

      summary = tf.compat.v1.Summary(
          value=[tf.compat.v1.Summary.Value(tag=tag, histo=hist)])
      self.writer.add_summary(summary, self.step)
    except Exception as e:  # pylint: disable=broad-except
      qj(e, 'Summ.hist caught exception. Ignoring.')

  def text(self, tag, value):
    """Save a text summary."""
    text = tf.compat.v1.make_tensor_proto(value, dtype=tf.string)
    meta = tf.compat.v1.SummaryMetadata()
    meta.plugin_data.plugin_name = 'text'

    summary = tf.compat.v1.Summary(
        value=[tf.compat.v1.Summary.Value(tag=tag, metadata=meta, tensor=text)])
    self.writer.add_summary(summary, self.step)

  def flush(self):
    self.writer.flush()

  def close(self):
    self.writer.close()


def image_strip_summary(name, images, max_length=100, max_batch=10):
  """Create an image summary that places frames of a video tensor side by side.

  Args:
    name: Name tag of the summary.
    images: Tensor with the dimensions batch, time, height, width, channels.
    max_length: Maximum number of frames per sequence to include.
    max_batch: Maximum number of sequences to include.

  Returns:
    Summary string tensor.
  """
  if max_batch:
    images = images[:max_batch]
  if max_length:
    images = images[:, :max_length]
  if images.dtype == tf.uint8:
    images = tf.cast(images, tf.float32) / 255.0
  length, width = tf.shape(images)[1], tf.shape(images)[3]
  channels = tf.shape(images)[-1]
  images = tf.transpose(images, [0, 2, 1, 3, 4])
  images = tf.reshape(images, [1, -1, length * width, channels])
  images = tf.clip_by_value(images, 0., 1.)
  return tf.summary.image(name, images)


def replay_summary(name, step=None, reshape=False, order_frame_stack=False,
                   image_summary_interval=10000, has_augmentations=False,
                   frame_stack=3):
  """Make a summary from a batch of replay buffer sequences."""
  if step is None:
    step = tf.Variable(
        0, dtype=tf.int64, trainable=False, name=name + '_step')
    auto_step = True
  else:
    auto_step = False

  def fn(input_obs, meta):
    """Actual summary function."""
    if image_summary_interval > 0:
      with tf.compat.v2.summary.record_if(
          lambda: tf.math.equal(step % image_summary_interval, 0)):
        obs = input_obs[0] if has_augmentations else input_obs
        if hasattr(obs, 'observation'):
          images = obs.observation['pixels']
        else:
          images = obs
        if reshape and frame_stack > 1:
          images = tf.reshape(
              images,
              [-1 if x is None else x for x in
               images.shape.as_list()[:-1]
               + [frame_stack, images.shape.as_list()[-1] // frame_stack]])
          images = tf.transpose(images, [0, 1, 4, 2, 3, 5])
          images = tf.reshape(
              images,
              [-1 if x is None else x for x in  # pylint: disable=g-complex-comprehension
               images.shape.as_list()[:1]
               + [images.shape.as_list()[1] * frame_stack]
               + images.shape.as_list()[3:]])
        elif order_frame_stack and frame_stack > 1:
          images = tf.reshape(
              images,
              [-1 if x is None else x for x in
               images.shape.as_list()[:-1]
               + [frame_stack, images.shape.as_list()[-1] // frame_stack]])
          front = images[:, :, ..., 0, :]
          back = images[:, -1, ..., 1:, :]
          back = tf.transpose(back, [0, 3, 1, 2, 4])
          images = tf.concat([front, back], axis=1)

        tf.summary.experimental.set_step(step.read_value())
        image_strip_summary(name, images, max_length=100, max_batch=10)
      # TODO(iansf): To enable showing the augmentation summaries, get rid of
      #              the 'and not auto_step' part of the condition.
      if has_augmentations and not auto_step:
        augmentations = input_obs[1]
        for i, f in enumerate(augmentations):
          f = tf.reshape(f, [
              -1 if x is None else x for x in f.shape.as_list()[:-1] +
              [frame_stack, f.shape.as_list()[-1] // frame_stack]
          ])
          f = tf.transpose(f, [0, 3, 1, 2, 4])
          image_strip_summary(
              name + '_aug{}'.format(i), f, max_length=100, max_batch=10)
      if auto_step:
        step.assign_add(1)
    return input_obs, meta
  return fn
