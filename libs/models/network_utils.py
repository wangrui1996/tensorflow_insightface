"""Core Keras layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.keras import layers

def Linear(x, config, group = False, num_filter=1, kernel_size=(1, 1), stride=(1, 1), padding="same", name=None, suffix=''):
    conv_name = '%s%s_conv2d' %(name, suffix)
    if group:
        x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding=padding, use_bias=False, name=conv_name)(x)
    else:
        x = layers.Conv2D(filters=num_filter, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=False, name=conv_name)(x)
    x = layers.BatchNormalization(momentum=config["bn_mom"], name='%s%s_batchnorm' %(name, suffix))(x)
    return x

def get_fc1(last_conv, config):

    if config["fc_type"] == "GDC":
        x = Linear(last_conv, config, group=True, kernel_size=(7, 7), padding="valid",
                           stride=(1, 1), name="conv_6dw7_7")
        x = layers.Flatten()(x)
        x = layers.Dense(config["embed_size"], name="pre_fc1")(x)
        x = layers.BatchNormalization(momentum=config["bn_mom"], scale=False,epsilon=2e-5, name='fc1')(x)
    return x

from tensorflow.python.eager import context
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import common_shapes
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import standard_ops
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import initializers
from tensorflow.python.keras import constraints
from tensorflow.python.keras import activations
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.keras.utils import tf_utils
@keras_export('keras.layers.Matmul')
class Matmul(Layer):
    def __init__(self, units, s=1, **kwargs):
        super(Matmul, self).__init__(**kwargs)
        self.units = units
        self.s = s

    def build(self, input_shape):
        #super(Matmul, self).build(input_shape[0])
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.kernel = self.add_weight(name='W',
                                shape=(last_dim, self.units),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=None)

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs)
        # normalize feature
        inputs = K.l2_normalize(inputs, axis=1)
        inputs = inputs*self.s
        # normalize weights
        W = K.l2_normalize(self.kernel, axis=0)
        # clip logits to prevent zero divi,sion when backward
        outputs = gen_math_ops.mat_mul(inputs, W)
        #out = tf.matmul(x, W)

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.units)

@keras_export('keras.layers. IdentityMatrixMul')
class IdentityMatrixMul(Layer):
  """Just your regular densely-connected NN layer.

  `Dense` implements the operation:
  `output = activation(dot(input, kernel) + bias)`
  where `activation` is the element-wise activation function
  passed as the `activation` argument, `kernel` is a weights matrix
  created by the layer, and `bias` is a bias vector created by the layer
  (only applicable if `use_bias` is `True`).

  Note: If the input to the layer has a rank greater than 2, then
  it is flattened prior to the initial dot product with `kernel`.

  Example:

  ```python
  # as first layer in a sequential model:
  model = Sequential()
  model.add(Dense(32, input_shape=(16,)))
  # now the model will take as input arrays of shape (*, 16)
  # and output arrays of shape (*, 32)

  # after the first layer, you don't need to specify
  # the size of the input anymore:
  model.add(Dense(32))
  ```

  Arguments:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation")..
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.

  Input shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`.
    The most common situation would be
    a 2D input with shape `(batch_size, input_dim)`.

  Output shape:
    N-D tensor with shape: `(batch_size, ..., units)`.
    For instance, for a 2D input with shape `(batch_size, input_dim)`,
    the output would have shape `(batch_size, units)`.
  """

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               l2_inputs=False,
               l2_weights=False,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super(IdentityMatrixMul, self).__init__(
        activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
    self.units = int(units)
    self.l2_inputs = l2_inputs
    self.l2_weights = l2_weights
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.supports_masking = True
    self.input_spec = InputSpec(min_ndim=2)

  def build(self, input_shape):
    dtype = dtypes.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))
    input_shape = tensor_shape.TensorShape(input_shape)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    last_dim = tensor_shape.dimension_value(input_shape[-1])
    self.input_spec = InputSpec(min_ndim=2,
                                axes={-1: last_dim})
    self.kernel = self.add_weight(
        'kernel',
        shape=[last_dim, self.units],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=[self.units,],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs_):
    inputs = ops.convert_to_tensor(inputs_)
    if self.l2_inputs:
        inputs = K.l2_normalize(inputs, axis=1)


    rank = common_shapes.rank(inputs)
    if rank > 2:
      # Broadcasting is required for the inputs.
      if self.l2_weights:
          kernel = K.l2_normalize(self.kernel, 0)
      else:
          kernel = self.kernel
      outputs = standard_ops.tensordot(inputs, kernel, [[rank - 1], [0]])
      # Reshape the output back to the original ndim of the input.
      if not context.executing_eagerly():
        shape = inputs.shape.as_list()
        output_shape = shape[:-1] + [self.units]
        outputs.set_shape(output_shape)
    else:
      # Cast the inputs to self.dtype, which is the variable dtype. We do not
      # cast if `should_cast_variables` is True, as in that case the variable
      # will be automatically casted to inputs.dtype.
      if not self._mixed_precision_policy.should_cast_variables:
        inputs = math_ops.cast(inputs, self.dtype)
      if self.l2_weights:
          kernel = K.l2_normalize(self.kernel, 0)
      else:
          kernel = self.kernel
      outputs = gen_math_ops.mat_mul(inputs, kernel)
    if self.use_bias:
      outputs = nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = {
        'units': self.units,
        'l2_inputs': False,
        'l2_weights': False,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    base_config = super(IdentityMatrixMul, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
