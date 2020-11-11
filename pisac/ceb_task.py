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
r"""Implementation of Conditional Entropy Bottleneck (CEB).

Fischer, I. "The conditional entropy bottleneck", 2020
https://arxiv.org/abs/2002.05379

The contrastive version of CEB consists of 3 components:
  1. Forward encoder e_zx
  2. Backward encoder b_zy
  3. CatGen
"""

import gin
from pstar import pdict
from pstar import plist

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tf_agents.utils import eager_utils

tfd = tfp.distributions


@gin.configurable
class CEB(tf.Module):
  """The Conditional Entropy Bottleneck."""

  def __init__(self,
               beta_fn=1.0,
               loss_weight=1.0,
               forward_smoothing_weight=0.6,
               smooth_mode='b',
               generative_ratio=0.0,
               generative_items=None,
               temperature=1.0,
               ood_log_density=None,
               img_pred_summary_fn=None,
               debug_summaries=True,
               step_counter=None,
               name='CEB'):
    """Create a CEB module.

    Args:
      beta_fn: A beta value schedule function.
      loss_weight: CEB loss weight.
      forward_smoothing_weight: forward smoothing weight.
      smooth_mode: 'b' for using b(z|y1), 'bc' for using b(z|y1) and c(y1|z),
        None for not doing smoothing.
      generative_ratio: Float between 0 and 1. If 0, only use the catgen loss.
        If 1, only use the predictive loss. Otherwise, mix between the two.
      generative_items: Number of items to predict with generative decoders.
        None (the default) means predict all y_target0 items. An int predicts
        (up to) that many items starting at the beginning of the list of
        y_targets0.
      temperature: Softmax temperature
      ood_log_density: Optional uniform log_density added to catgen logits that
        forces the confidence of the model to be above a certain level if
        possible, which tends to prevent the latent space from getting too
        dispersed. None or >= 0 disables.
      img_pred_summary_fn: Optional image prediction summary function.
      debug_summaries: Enable debug summaries.
      step_counter: training step counter.
      name: A string representing name of the module.
    """
    self.beta_fn = beta_fn
    self.loss_w = loss_weight
    self.fw_smooth_w = forward_smoothing_weight
    self.smooth_mode = smooth_mode
    self.generative_ratio = generative_ratio
    self.generative_items = generative_items
    self.tau = temperature
    self.ood_log_density = ood_log_density
    self.img_pred_summary_fn = img_pred_summary_fn
    self.debug_summaries = debug_summaries
    self.step_counter = step_counter
    self.model_name = name

  def decoder_catgen_dist(self, critic_dist, z):
    """Contrastive CatGen decoder.

    Args:
      critic_dist: Critic distribution with shape [B T (Z)]
      z: representaiton vector with shape [B T Z].

    Returns:
      A pdict containing CatGen categorical distribution and log(c(y|z)).
    """
    dist = pdict()

    def ood_logits(logits, ood_log_density):
      logits = tf.concat(
          [logits, tf.ones_like(logits[..., :1]) * ood_log_density], axis=-1)
      return logits

    def log_prob():
      batch_shape = critic_dist.batch_shape
      batch_size, seq_size = batch_shape[0], batch_shape[1]
      z_dim = z.shape[-1]
      # z_squash: [BxT Z]
      z_squash = tf.reshape(z, shape=(batch_size*seq_size, z_dim))
      # logits: [BxT B T], z_squash -> [BxT 1 1 Z]
      logits = critic_dist.log_prob(z_squash[..., None, None, :])
      # logits: [B T BxT]
      logits = tf.reshape(
          logits, shape=(batch_size, seq_size, batch_size * seq_size))

      if self.ood_log_density is not None and self.ood_log_density < 0:
        logits = ood_logits(logits, self.ood_log_density)

      # temperature
      if self.tau != 1.0:
        logits = logits / self.tau

      # dist.cat_dist: [B T (BxT)]
      dist.cat_dist = tfd.Categorical(logits=logits)

      inds = tf.range(batch_size * seq_size)
      inds = tf.reshape(inds, shape=(batch_size, seq_size))
      # log_probs: [B T]
      log_probs = dist.cat_dist.log_prob(inds)
      mi_upper_bound = tf.math.log(tf.cast(batch_size * seq_size, tf.float32))
      log_probs = log_probs + mi_upper_bound
      return log_probs

    dist.log_prob = log_prob
    return dist

  def loss(self,
           zx0,
           e_zx_0,
           b_zy_0,
           b_zy_1=None,
           y_preds0=None,
           y_targets0=None):
    """CEB loss.

    Args:
      zx0: representation vector sampled from e_zx_0 with shape [B T Z].
      e_zx_0: e(z|x) with shape [B T (Z)].
      b_zy_0: b(z|y) with shape [B T (Z)].
      b_zy_1: Optional b(z|y_{t+1}) with shape [B T (Z)] for smoothing.
      y_preds0: Optional predicted y for generative objectives.
      y_targets0: Optional target y for generative objectives.

    Returns:
      CEB loss.
    """
    beta = self.beta_fn()
    log_ezx_0 = e_zx_0.log_prob(zx0)
    log_bzy_0 = b_zy_0.log_prob(zx0)
    i_xz_y_0 = log_ezx_0 - log_bzy_0
    c_yz_0 = self.decoder_catgen_dist(b_zy_0, zx0)
    i_yz_0 = c_yz_0.log_prob()
    is_pure_contrastive = (y_preds0 is None) or (self.generative_ratio == 0.0)

    # loss: [B T]
    if is_pure_contrastive:
      loss = self.loss_w * (beta * i_xz_y_0 - i_yz_0)
    else:
      log_cyz0 = 0.0
      log_cyz0_all = []
      y_preds0 = y_preds0[:self.generative_items]
      y_targets0 = y_targets0[:self.generative_items]
      log_cyz0_all = y_preds0.apply(
          tf.losses.mean_squared_error, plist(y_targets0))
      # Handle unreduced pixel dimensions.
      log_cyz0_all[0] = tf.reduce_sum(log_cyz0_all[0], axis=(-1, -2))
      log_cyz0 = tf.reduce_sum(
          tf.concat(log_cyz0_all._[..., None], axis=-1), axis=-1)
      phi = self.generative_ratio
      loss = self.loss_w * (beta*i_xz_y_0 + phi*log_cyz0 - (1.0-phi)*i_yz_0)

    if self.smooth_mode in ['b', 'bc']:
      log_bzy_1 = b_zy_1.log_prob(zx0)
      i_xz_y_1 = log_ezx_0 - log_bzy_1
      loss += self.fw_smooth_w * beta * i_xz_y_1

    if self.smooth_mode == 'bc':
      c_yz_1 = self.decoder_catgen_dist(b_zy_1, zx0)
      i_yz_1 = c_yz_1.log_prob()
      loss -= self.fw_smooth_w * i_yz_1

    # Invalid transitions are prefiltered.
    loss = tf.reduce_mean(loss)

    with tf.name_scope(self.model_name):
      tf.summary.scalar(name='Beta', data=beta, step=self.step_counter)
      tf.summary.scalar(name='Loss', data=loss, step=self.step_counter)
      tf.summary.scalar(name='Iyz', data=tf.reduce_mean(i_yz_0),
                        step=self.step_counter)
      tf.summary.scalar(name='Ixz_y', data=tf.reduce_mean(i_xz_y_0),
                        step=self.step_counter)
      if self.smooth_mode in ['b', 'bc']:
        tf.summary.scalar(name='Ixz_y_1', data=tf.reduce_mean(i_xz_y_1),
                          step=self.step_counter)
      if self.smooth_mode == 'bc':
        tf.summary.scalar(name='Iyz_1', data=tf.reduce_mean(i_yz_1),
                          step=self.step_counter)
      if self.debug_summaries:
        tf.summary.scalar(name='Hz_x', data=tf.reduce_mean(-log_ezx_0),
                          step=self.step_counter)
        tf.summary.scalar(name='Hz_y', data=tf.reduce_mean(-log_bzy_0),
                          step=self.step_counter)
        tf.summary.histogram(name='zx0', data=zx0, step=self.step_counter)
        tf.summary.histogram('c_yz_0_logits', c_yz_0.cat_dist.logits,
                             step=self.step_counter)
      if not is_pure_contrastive:
        tf.summary.scalar('Hyz0', tf.reduce_mean(log_cyz0),
                          step=self.step_counter)
        for i, log_cyz in enumerate(log_cyz0_all):
          tf.summary.scalar('Hyz0_%s' % (['pixels', 'rewards'][i]),
                            tf.reduce_mean(log_cyz), step=self.step_counter)
        self.img_pred_summary_fn(obs=y_targets0[0], pred=y_preds0[0])
    return loss


@gin.configurable
class CEBTask(tf.Module):
  """CEB task."""

  def __init__(self,
               ceb,
               forward_enc,
               backward_enc,
               forward_head,
               backward_head,
               y_decoders=None,
               action_condition=True,
               backward_encode_rewards=True,
               learn_backward_enc=False,
               optimizer=None,
               grad_clip=None,
               global_step=None,
               name='CEBTask'):
    """Create a CEB task.

    Args:
      ceb: A CEB module.
      forward_enc: Forward encoder torso.
      backward_enc: Backward ecoder torso.
      forward_head: Forward encoder head.
      backward_head: Backward encoder head.
      y_decoders: Optional list of generative decoders of y values to maximize
        log-likelihood.
      action_condition: Forward encoder also encodes actions if True.
      backward_encode_rewards: Backward encode rewards if True.
      learn_backward_enc: Stop gradient to backward_enc if False.
      optimizer: CEB task optimizer.
      grad_clip: A scalar, gradient norm clip-off.
      global_step: Training step counter.
      name: A string representing name of the task module.
    """
    self.forward_enc = forward_enc
    self.forward_head = forward_head
    self.backward_enc = backward_enc
    self.backward_head = backward_head
    self.y_decoders = None if y_decoders is None else plist(y_decoders)
    self.action_condition = action_condition
    self.backward_encode_rewards = backward_encode_rewards
    self.learn_backward_enc = learn_backward_enc

    self.ceb = ceb
    self.optimizer = optimizer
    self.grad_clip = grad_clip
    self.global_step = global_step
    self.model_name = name

  def train(self, x0, a0, y0, y1, r0, r1, vars_to_train):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(vars_to_train)

      feat_x0, _ = self.forward_enc(x0, training=True)
      if self.action_condition:
        e_zx0_param, _ = self.forward_head([feat_x0, a0], training=True)
      else:
        e_zx0_param, _ = self.forward_head(feat_x0, training=True)
      e_zx0_loc, e_zx0_scale = e_zx0_param
      e_zx0 = tfd.MultivariateNormalDiag(loc=e_zx0_loc, scale_diag=e_zx0_scale)
      zx0 = e_zx0.sample()

      feat_y0, _ = self.backward_enc(y0, training=self.learn_backward_enc)
      if not self.learn_backward_enc:
        feat_y0 = tf.stop_gradient(feat_y0)
      if self.backward_encode_rewards:
        b_zy0_param, _ = self.backward_head([feat_y0, r0], training=True)
      else:
        b_zy0_param, _ = self.backward_head(feat_y0, training=True)
      b_zy0_loc, b_zy0_scale = b_zy0_param
      b_zy0 = tfd.MultivariateNormalDiag(loc=b_zy0_loc, scale_diag=b_zy0_scale)

      b_zy1 = None
      if self.ceb.smooth_mode is not None:
        feat_y1, _ = self.backward_enc(y1, training=self.learn_backward_enc)
        if not self.learn_backward_enc:
          feat_y1 = tf.stop_gradient(feat_y1)
        if self.backward_encode_rewards:
          b_zy1_param, _ = self.backward_head([feat_y1, r1], training=True)
        else:
          b_zy1_param, _ = self.backward_head(feat_y1, training=True)
        b_zy1_loc, b_zy1_scale = b_zy1_param
        b_zy1 = tfd.MultivariateNormalDiag(loc=b_zy1_loc,
                                           scale_diag=b_zy1_scale)

      if self.y_decoders is None:  # pure contrastive CEB
        loss = self.ceb.loss(zx0, e_zx0, b_zy0, b_zy1)
      else:  # CEB with generative objectives
        # y_targets0 = [y0, r0]
        y_targets0 = [tf.cast(y0, tf.float32) / 255.0, r0]
        y_preds0 = self.y_decoders(zx0, training=True)._[0]
        loss = self.ceb.loss(zx0, e_zx0, b_zy0, b_zy1, y_preds0, y_targets0)

    grads = tape.gradient(loss, vars_to_train)
    grads_and_vars = tuple(zip(grads, vars_to_train))
    if self.grad_clip is not None:
      grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                       self.grad_clip)
    self.optimizer.apply_gradients(grads_and_vars)
    return loss, feat_x0, zx0
