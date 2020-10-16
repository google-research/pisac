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
"""Schedule Utils."""

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp


tfd = tfp.distributions


def get_schedule_fn(base, sched=None, step=None):
  """Get a schedule function.

  Args:
    base: Base rate.
    sched: Optional `String` that describe a schedule.
    step: Optional `Tensor` with current step as a int.

  Returns:
    A schedule function.
  """
  if sched is not None:
    return lambda: base * schedule_from_str(sched, step)
  else:
    return lambda: base


def schedule_from_str(sched, current_step, default_scale=0.1,
                      default_count=5000, default_interp='lerp'):
  """Generic parameter schedule from a string.

  Args:
    sched: <optional_interp>@<step>:<optional_scale>:<optional_count>_<same>...
           E.g., '0:0:0_5:1_30_60_80' or 'berp@0:1:0_5:0.9_10:0.8_15:0.7'
    current_step: Tensor with current step as a int.
    default_scale: What to multiply the previous value by if the string doesn't
                   specify a value.
    default_count: How many steps to interpolate over if the string doesn't
                   specify a count.
    default_interp: 'lerp' for linear interpolation, 'berp' for beta
                    distribution interpolation, 'expd' for exponential decay.
                    Can be set with an initial <interp>@ at the beginning of
                    sched.

  Returns:
    A dynamic tensor that changes the parameter according to the schedule.
  """
  interp, _, sched = sched.partition('@')
  if not sched:
    sched = interp
    interp = default_interp
  default_interp = interp
  interp = globals()[interp]

  interps = []
  steps = []
  values = []
  counts = []
  for interp in sched.split('_'):
    interp, _, step = interp.partition('@')
    if not step:
      step = interp
      interp = default_interp
    step, _, value = step.partition(':')
    value, _, count = value.partition(':')
    interp = globals()[interp]
    step = int(step)
    value = float(value) if value else default_scale ** len(values)
    count = int(count) if count else default_count
    interps.append(interp)
    steps.append(step)
    values.append(value)
    counts.append(count)

  # Go from 0 to 1 over 5 epochs, then stay constant for 5 more epochs,
  # then go to 0.5 over 3 epochs:
  # <start_step>[:target_value[:number_of_steps]]
  # '5_10:0.5:3'
  # berp@1000_10000:0.3:0'
  # looks like epochs[0] shoud also be 0? 'berp@0:1.0:1000_10000:0.3:0'
  param = interps[0](current_step, 0, steps[0] + counts[0], 0.0, values[0])
  for interp, step, value, count in list(
      zip(interps, steps, values, counts))[1:]:
    param = tf.where(
        current_step < step,
        param,
        interp(current_step, step, step + count, param, value))
  return param


def expd(global_step, start_step, end_step, start_val, end_val, warmup=True):
  """Exponential decay interpolation."""
  if warmup and start_step == 0:
    return lerp(global_step, start_step, end_step, start_val, end_val)
  decay_steps = end_step - start_step
  decay_factor = end_val
  return tf.optimizers.schedules.ExponentialDecay(
      start_val, decay_steps, decay_factor, staircase=True)(
          global_step - start_step)


def subexpd_np(step, start_d, start_v, d_decay_rate, v_decay_rate, start_t=0,
               stair=False):
  """Sub-exponential decay function.

  Closed-form solution to the problem of doing an exponential decay of a value
  while with exponentially more steps between each decay.

  If d_decay_rate = v_decay_rate, this is (nearly) equivalent to
  InverseTimeDecay. As d_decay_rate -> 1, this becomes equivalent to
  ExponentialDecay.

  Args:
    step: current step or array of steps.
    start_d: initial duration between each decay step.
    start_v: initial value to decay.
    d_decay_rate: how quickly to decay the duration. Values less than 1 cause
      the duration to increase over time. Values greater than 1 cause the
      duration to decrease over time (corresponding to a super-exponential
      decay). Setting to 1 gives NaNs, but in the limit is equivalent to
      exponential decay.
    v_decay_rate: how quickly to decay the value. Values less than 1 decay the
      initial value. Values greater than 1 grow the initial value.
    start_t: time offset at which to start the decay.
    stair: if True, gives a stairstepped output. If False, gives a smooth
      output.

  Returns:
    Sub-exponentially-decayed value or values (if step was an array).
  """
  # The code can be modified to use numpy by removing the following line:
  np = tf.math
  step -= start_t
  exp = (
      np.log(-np.log(d_decay_rate) * step / start_d + 1)
      / -np.log(d_decay_rate))
  if stair:
    exp = np.floor(exp)
  return start_v * v_decay_rate ** exp


def subexpd(global_step, start_step, end_step, start_val, end_val,
            warmup=True, stair=True):
  """Sub-exponential decay function. Duration decay is sqrt(decay)."""
  if warmup and start_step == 0:
    return lerp(global_step, start_step, end_step, start_val, end_val)
  decay_steps = tf.cast(end_step - start_step, tf.float32)
  decay_factor = tf.cast(end_val, tf.float32)
  d_decay_factor = tf.cast(tf.sqrt(decay_factor), tf.float32)
  step = tf.cast(global_step - start_step, tf.float32)
  return subexpd_np(
      step, decay_steps, start_val, d_decay_factor, decay_factor, stair=stair)


def berp(global_step, start_step, end_step, start_val, end_val, alpha=5):
  """Beta interpolation."""
  beta_dist = tfd.Beta(alpha, alpha)
  mode = beta_dist.mode()
  interp = (tf.cast(global_step - start_step, tf.float32)
            / tf.cast(end_step - start_step, tf.float32))
  interp = tf.maximum(0.0, tf.minimum(1.0, interp))
  interp = tf.where(tf.math.is_nan(interp), tf.zeros_like(interp), interp)
  interp *= mode
  val = beta_dist.prob(interp)
  val /= beta_dist.prob(mode)
  val *= (end_val - start_val)
  val += start_val
  return val


def lerp(global_step, start_step, end_step, start_val, end_val):
  """Linear interpolation."""
  interp = (tf.cast(global_step - start_step, tf.float32)
            / tf.cast(end_step - start_step, tf.float32))
  interp = tf.where(tf.math.is_nan(interp), tf.zeros_like(interp), interp)
  interp = tf.maximum(0.0, tf.minimum(1.0, interp))
  return start_val * (1.0 - interp) + end_val * interp
