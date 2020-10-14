"""Project inputs to a tanh-squashed MultivariateNormalDiag distribution."""

import gin
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.distributions import utils as distribution_utils
from tf_agents.networks import network
from tf_agents.networks import utils as network_utils
from tf_agents.specs import distribution_spec
from tf_agents.specs import tensor_spec


@gin.configurable
class TanhNormalProjectionNetwork(network.DistributionNetwork):
  """Generates a tanh-squashed MultivariateNormalDiag distribution."""

  def __init__(self,
               sample_spec,
               activation_fn=None,
               kernel_initializer=None,
               std_transform=tf.exp,
               min_std=None,
               max_std=None,
               name='TanhNormalProjectionNetwork'):
    """Creates an instance of TanhNormalProjectionNetwork.

    Args:
      sample_spec: A `tensor_spec.BoundedTensorSpec` detailing the shape and
        dtypes of samples pulled from the output distribution.
      activation_fn: Activation function to use in dense layer.
      kernel_initializer: Initializer to use for the kernels of the conv and
        dense layers. If none is provided a default glorot_uniform
      std_transform: Transformation function to apply to the stddevs.
      min_std: Minimum std.
      max_std: Maximum std.
      name: A string representing name of the network.
    """
    if len(tf.nest.flatten(sample_spec)) != 1:
      raise ValueError('Tanh Normal Projection network only supports single'
                       ' spec samples.')
    output_spec = self._output_distribution_spec(sample_spec, name)
    super(TanhNormalProjectionNetwork, self).__init__(
        # We don't need these, but base class requires them.
        input_tensor_spec=None,
        state_spec=(),
        output_spec=output_spec,
        name=name)

    self._sample_spec = sample_spec
    self._std_transform = std_transform
    self._min_std = min_std
    self._max_std = max_std

    if kernel_initializer is None:
      kernel_initializer = 'glorot_uniform'

    self._projection_layer = tf.keras.layers.Dense(
        sample_spec.shape.num_elements() * 2,
        activation=activation_fn,
        kernel_initializer=kernel_initializer,
        name='projection_layer')

  def _output_distribution_spec(self, sample_spec, network_name):
    input_param_shapes = {
        'loc': sample_spec.shape,
        'scale_diag': sample_spec.shape
    }
    input_param_spec = {  # pylint: disable=g-complex-comprehension
        name: tensor_spec.TensorSpec(
            shape=shape,
            dtype=sample_spec.dtype,
            name=network_name + '_' + name)
        for name, shape in input_param_shapes.items()
    }

    def distribution_builder(*args, **kwargs):
      distribution = tfp.distributions.MultivariateNormalDiag(*args, **kwargs)
      return distribution_utils.scale_distribution_to_spec(
          distribution, sample_spec)

    return distribution_spec.DistributionSpec(
        distribution_builder, input_param_spec, sample_spec=sample_spec)

  def call(self, inputs, outer_rank, training=False, mask=None):
    if inputs.dtype != self._sample_spec.dtype:
      raise ValueError('Inputs to TanhNormalProjectionNetwork must match the '
                       'sample_spec.dtype.')

    if mask is not None:
      raise NotImplementedError(
          'TanhNormalProjectionNetwork does not yet implement action masking; '
          'got mask={}'.format(mask))

    # outer_rank is needed because the projection is not done on the raw
    # observations so getting the outer rank is hard as there is no spec to
    # compare to.
    batch_squash = network_utils.BatchSquash(outer_rank)
    inputs = batch_squash.flatten(inputs)

    means_and_stds = self._projection_layer(inputs, training=training)
    means, stds = tf.split(means_and_stds, num_or_size_splits=2, axis=-1)
    means = tf.reshape(means, [-1] + self._sample_spec.shape.as_list())
    means = tf.cast(means, self._sample_spec.dtype)

    if self._std_transform is not None:
      stds = self._std_transform(stds)
    if self._min_std is not None:
      stds = tf.maximum(stds, self._min_std)
    if self._max_std is not None:
      stds = tf.minimum(stds, self._max_std)
    stds = tf.cast(stds, self._sample_spec.dtype)

    means = batch_squash.unflatten(means)
    stds = batch_squash.unflatten(stds)

    return self.output_spec.build_distribution(loc=means, scale_diag=stds), ()
