"""Metrics Utils."""

import collections

import tensorflow as tf

from tf_agents.drivers import dynamic_episode_driver
from tf_agents.metrics import tf_metric
from tf_agents.utils import common


class TFDeque(object):
  """Deque backed by tf.Variable storage."""

  def __init__(self, max_len, dtype, name='TFDeque'):
    shape = (max_len,)
    self._dtype = dtype
    self._max_len = tf.convert_to_tensor(max_len, dtype=tf.int32)
    self._buffer = common.create_variable(
        initial_value=0, dtype=dtype, shape=shape, name=name + 'Vars')

    self._head = common.create_variable(
        initial_value=0, dtype=tf.int32, shape=(), name=name + 'Head')

  @property
  def data(self):
    return self._buffer[:self.length]

  @common.function(autograph=True)
  def extend(self, value):
    for v in value:
      self.add(v)

  @common.function(autograph=True)
  def add(self, value):
    position = tf.math.mod(self._head, self._max_len)
    self._buffer.scatter_update(tf.IndexedSlices(value, position))
    self._head.assign_add(1)

  @property
  def length(self):
    return tf.minimum(self._head, self._max_len)

  @common.function
  def clear(self):
    self._head.assign(0)
    self._buffer.assign(tf.zeros_like(self._buffer))

  @common.function(autograph=True)
  def mean(self):
    if tf.equal(self._head, 0):
      return tf.zeros((), dtype=self._dtype)
    return tf.math.reduce_mean(self._buffer[:self.length])

  @common.function(autograph=True)
  def stddev(self):
    if tf.equal(self._head, 0):
      return tf.zeros((), dtype=self._dtype)
    return tf.math.reduce_std(self._buffer[:self.length])


class ReturnStddevMetric(tf_metric.TFStepMetric):
  """Metric to compute the return standard deviation."""

  def __init__(self,
               name='ReturnStddev',
               prefix='Metrics',
               dtype=tf.float32,
               batch_size=1,
               buffer_size=10):
    super(ReturnStddevMetric, self).__init__(name=name, prefix=prefix)
    self._buffer = TFDeque(buffer_size, dtype)
    self._dtype = dtype
    self._return_accumulator = common.create_variable(
        initial_value=0, dtype=dtype, shape=(batch_size,), name='Accumulator')

  @common.function(autograph=True)
  def call(self, trajectory):
    # Zero out batch indices where a new episode is starting.
    self._return_accumulator.assign(
        tf.where(trajectory.is_first(), tf.zeros_like(self._return_accumulator),
                 self._return_accumulator))

    # Update accumulator with received rewards.
    self._return_accumulator.assign_add(trajectory.reward)

    # Add final returns to buffer.
    last_episode_indices = tf.squeeze(tf.where(trajectory.is_last()), axis=-1)
    for indx in last_episode_indices:
      self._buffer.add(self._return_accumulator[indx])

    return trajectory

  def result(self):
    return self._buffer.stddev()

  @common.function
  def reset(self):
    self._buffer.clear()
    self._return_accumulator.assign(tf.zeros_like(self._return_accumulator))


class ReturnHistogram(tf_metric.TFHistogramStepMetric):
  """Metric to compute the frequency of each action chosen."""

  def __init__(self,
               name='ReturnHistogram',
               dtype=tf.float32,
               batch_size=1,
               buffer_size=10):
    super(ReturnHistogram, self).__init__(name=name)
    self._buffer = TFDeque(buffer_size, dtype)
    self._dtype = dtype
    self._return_accumulator = common.create_variable(
        initial_value=0, dtype=dtype, shape=(batch_size,), name='Accumulator')

  @common.function(autograph=True)
  def call(self, trajectory):
    # Zero out batch indices where a new episode is starting.
    self._return_accumulator.assign(
        tf.where(trajectory.is_first(), tf.zeros_like(self._return_accumulator),
                 self._return_accumulator))

    # Update accumulator with received rewards.
    self._return_accumulator.assign_add(trajectory.reward)

    # Add final returns to buffer.
    last_episode_indices = tf.squeeze(tf.where(trajectory.is_last()), axis=-1)
    for indx in last_episode_indices:
      self._buffer.add(self._return_accumulator[indx])

    return trajectory

  # @common.function
  def result(self):
    return self._buffer.data

  @common.function
  def reset(self):
    self._buffer.clear()
    self._return_accumulator.assign(tf.zeros_like(self._return_accumulator))


def eager_compute(metrics,
                  environment,
                  policy,
                  histograms=None,
                  num_episodes=1,
                  train_step=None,
                  summary_writer=None,
                  summary_prefix='',
                  use_function=True):
  """Compute metrics using `policy` on the `environment`.

  *NOTE*: Because placeholders are not compatible with Eager mode we can not use
  python policies. Because we use tf_policies we need the environment time_steps
  to be tensors making it easier to use a tf_env for evaluations. Otherwise this
  method mirrors `compute` directly.

  Args:
    metrics: List of metrics to compute.
    environment: tf_environment instance.
    policy: tf_policy instance used to step the environment.
    histograms: (Optional) List of histograms to compute.
    num_episodes: Number of episodes to compute the metrics over.
    train_step: An optional step to write summaries against.
    summary_writer: An optional writer for generating metric summaries.
    summary_prefix: An optional prefix scope for metric summaries.
    use_function: Option to enable use of `tf.function` when collecting the
      metrics.
  Returns:
    A dictionary of results {metric_name: metric_value}
  """
  if histograms is None:
    histograms = []
  for metric in metrics:
    metric.reset()
  for histogram in histograms:
    histogram.reset()

  time_step = environment.reset()
  policy_state = policy.get_initial_state(environment.batch_size)

  driver = dynamic_episode_driver.DynamicEpisodeDriver(
      environment,
      policy,
      observers=metrics+histograms,
      num_episodes=num_episodes)
  if use_function:
    common.function(driver.run)(time_step, policy_state)
  else:
    driver.run(time_step, policy_state)

  results = [(metric.name, metric.result()) for metric in metrics]
  if train_step is not None and summary_writer:
    with summary_writer.as_default():
      for m in metrics:
        tag = common.join_scope(summary_prefix, m.name)
        tf.compat.v2.summary.scalar(name=tag, data=m.result(), step=train_step)
      for h in histograms:
        tag = common.join_scope(summary_prefix, h.name)
        tf.compat.v2.summary.histogram(
            name=tag, data=h.result(), step=train_step)
  return collections.OrderedDict(results)
