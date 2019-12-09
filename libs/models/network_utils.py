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

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import regularizers
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
