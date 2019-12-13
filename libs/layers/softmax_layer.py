from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn
from tensorflow.python.keras import backend
import tensorflow as tf
@keras_export('keras.layers.Fsoftmax')
class Logits(Layer):
  """Softmax activation function.

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as the input.

  Arguments:
    axis: Integer, axis along which the softmax normalization is applied.
  """

  def __init__(self, class_num, normalize_input=False, normalize_weights=False, use_bias=True, **kwargs):
      super(Logits, self).__init__(**kwargs)
      self.class_num = class_num
      self.normalize_input = normalize_input
      self.normalize_weights = normalize_weights
      self.use_bias = use_bias

  def build(self, input_shape):
      input_shape = tensor_shape.TensorShape(input_shape)
      if tensor_shape.dimension_value(input_shape[-1]) is None:
          raise ValueError('The last dimension of the inputs to `Dense` '
                           'should be defined. Found `None`.')
      last_dim = tensor_shape.dimension_value(input_shape[-1])
      self.kernel = self.add_weight(name='kernel',
                                    shape=(last_dim, self.class_num),
                                    initializer='glorot_normal',
                                    trainable=True)
      if self.use_bias:
          self.bias = self.add_weight(
              'bias',
              shape=[self.class_num,],
              initializer='glorot_normal',
              trainable=True)
      else:
          self.bias = None
      self.build = True

  def call(self, inputs, **kwargs):
      layer_input = inputs
      if self.normalize_input:
          layer_input = tf.nn.l2_normalize(inputs, axis=1, name="normed_embeddings")
      weights = self.weights
      if self.normalize_weights:
          weights = tf.nn.l2_normalize(self.kernel, axis=0, name="normed_weights")
      outputs = tf.matmul(layer_input, weights, name='cos_t')
      if self.use_bias:
          outputs = nn.bias_add(outputs, self.bias)
      return outputs

  def get_config(self):
    config = {'class_num': self.class_num, 'use_bias': self.use_bias, 'normalize_input': self.normalize_input, 'normalize_weights': self.normalize_weights}
    base_config = super(Logits, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def compute_output_shape(self, input_shape):
      input_shape = tensor_shape.TensorShape(input_shape)
      input_shape = input_shape.with_rank_at_least(2)
      if tensor_shape.dimension_value(input_shape[-1]) is None:
          raise ValueError(
              'The innermost dimension of input_shape must be defined, but saw: %s'
              % input_shape)
      return input_shape[:-1].concatenate(self.class_num)
