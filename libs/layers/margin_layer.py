from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
import tensorflow as tf
@keras_export('keras.layers.Margin')
class Margin(Layer):
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

  def __init__(self, class_num, s=1, m1=1.0, m2=0.0, m3=0.0,**kwargs):
      super(Margin, self).__init__(**kwargs)
      self.class_num = class_num
      self.s = s
      self.m1 = m1
      self.m2 = m2
      self.m3 = m3

  def build(self, input_shape):
      input_shape = input_shape[0]
      input_shape = tensor_shape.TensorShape(input_shape)
      if tensor_shape.dimension_value(input_shape[-1]) is None:
          raise ValueError('The last dimension of the inputs to `Dense` '
                           'should be defined. Found `None`.')
      last_dim = tensor_shape.dimension_value(input_shape[-1])
      self.kernel = self.add_weight(name='kernel',
                                    shape=(last_dim, self.class_num),
                                    initializer='glorot_normal',
                                    trainable=True)
      self.build = True

  def call(self, inputs, **kwargs):
      s, m1, m2, m3 = (self.s, self.m1, self.m2, self.m3)

      embeddings = tf.nn.l2_normalize(inputs[0], axis=1, name="normed_embeddings")
      weights = tf.nn.l2_normalize(self.kernel, axis=0, name="normed_weights")
      embeddings = embeddings*s
      fc7 = tf.matmul(embeddings, weights, name='cos_t')
      if m1 == 1.0 and m2 == 0.0:
        s_m = s*m3
        gt_one_hot = inputs[1] * s_m
        output = fc7 - gt_one_hot
      else:
          zy = fc7
          cos_t = zy/s
          cos_t = backend.clip(cos_t, -1.0 + backend.epsilon(), 1.0 - backend.epsilon())
          t = tf.math.acos(cos_t)
          if m1 != 1.0:
              t = t * m1
          if m2 > 0.0:
              t = t + m2
          body = backend.cos(t)
          if m3 > 0.0:
              body = body - m3

          new_zy = body * s
          output = fc7 * (1-inputs[1]) + inputs[1] * new_zy

      return output

  def get_config(self):
    config = {'class_num': self.class_num, 's': self.s, 'm1': self.m1, 'm2': self.m2, 'm3': self.m3}
    base_config = super(Margin, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def compute_output_shape(self, input_shape):
      input_shape = tensor_shape.TensorShape(input_shape)
      input_shape = input_shape.with_rank_at_least(2)
      if tensor_shape.dimension_value(input_shape[-1]) is None:
          raise ValueError(
              'The innermost dimension of input_shape must be defined, but saw: %s'
              % input_shape)
      return input_shape[:-1].concatenate(self.class_num)
