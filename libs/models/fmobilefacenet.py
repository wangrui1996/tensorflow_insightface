import os

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import models

from libs.models.network_utils import get_fc1

def Conv(x, config, num_filter=1, kernel_size=(1, 1), stride=(1, 1), padding="same", group=False, name=None, suffix=''):
    conv_name = "%s%s_conv2d"%(name, suffix)
    if group:
        x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding=padding, use_bias=False, name=conv_name)(x)
    else:
        x = layers.Conv2D(filters=num_filter, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=False, name=conv_name)(x)
    x = layers.BatchNormalization(momentum=config["bn_mom"], name='%s%s_batchnorm' %(name, suffix))(x)
    if config["net_act"] == "relu":
        activation = tf.nn.relu_layer
    elif config["net_act"] == "prelu":
        activation = tf.nn.leaky_relu
    else:
        activation= None
        print("can not get activation layer from {}".format(config["net_act"]))
        exit(0)
    x = layers.Activation(activation=activation, name='%s%s_relu' %(name, suffix))(x)
    return x

def Linear(x, config, num_filter=1, kernel_size=(1, 1), stride=(1, 1), padding="same", name=None, suffix=''):
    x = layers.Conv2D(filters=num_filter, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=False, name='%s%s_conv2d' %(name, suffix))(x)
    x = layers.BatchNormalization(momentum=config["bn_mom"], name='%s%s_batchnorm' %(name, suffix))(x)
    return x

def DResidual(x, config, num_out=1, kernel_size=(3, 3), stride=(2, 2), padding="same", num_group=1, name=None, suffix=''):
    x = Conv(x, config, num_filter=num_group, kernel_size=(1,1), padding="valid", stride=(1,1), name='%s%s_conv_sep' %(name, suffix))
    x = Conv(x, config, group=True, kernel_size=kernel_size, padding=padding, stride=stride, name='%s%s_conv_dw' %(name, suffix))
    x = Linear(x, config, num_filter=num_out, kernel_size=(1, 1), padding="same", stride=(1, 1), name='%s%s_conv_proj' %(name, suffix))
    return x

def Residual(x, config, num_block=1, num_out=1, kernel_size=(3, 3), stride=(1, 1), padding="same", num_group=1, name=None, suffix=''):
    identity=x
    for i in range(num_block):
        shortcut=identity
        conv=DResidual(identity, config, num_out=num_out, kernel_size=kernel_size, stride=stride, padding=padding, num_group=num_group, name='%s%s_block' %(name, suffix), suffix='%d'%i)
        identity = layers.Add()([conv, shortcut])
    return identity
import tensorflow as tf
import math
import numpy as np
from tensorflow.python.keras import layers
import tensorflow.keras.backend as K
from tensorflow.python import keras
from tensorflow.python.keras import regularizers

def get_call_func(y_true, y_pred, config):
    is_softmax = True

    if config['loss_type'] == "softmax":
        logits = layers.Dense(config['class_num'], use_bias=config["fc7_use_bias"], name="fc7")(y_pred)
    elif config['loss_type'] == 'margin':
        logits = margin_softmax(y_pred, y_true, config)
    else:
        raise ValueError('Invalid loss type.')
    return logits


def margin_softmax(embedding, y_true, config):

    s = config["loss_s"]

    nembedding = embedding*s
    fc7 = layers.Dense(units=config["class_num"], use_bias=False, kernel_regularizer=K.l2_normalize, name="cos0")(nembedding)

    if config["loss_m1"] == 1.0 and config["loss_m2"] == 0.0:
        s_m = s * config["loss_m3"]
        gt_one_hot = tf.one_hot(y_true, depth=config["class_num"], on_value=s_m, off_value=0.0)
        output = fc7 - gt_one_hot
    else:
        zy = fc7
        cos_t = zy / s
        t = tf.math.acos(cos_t)
        if config["loss_m1"] != 1.0:
            t = t * config["loss_m1"]
        if config["loss_m2"] > 0.0:
            t = t + config["loss_m2"]
        body = K.cos(t)
        if config["loss_m3"] > 0.0:
            body = body - config["loss_m3"]
        new_zy = body * s
        bool_one_hot = tf.one_hot(y_true, depth=config["classes"], on_value=True, off_value=False)
        output = tf.where(bool_one_hot, new_zy, fc7)

    return output

def get_network(config, y_true=None, is_train=False):
    blocks = config["net_blocks"]

    img_input = layers.Input(shape=config["input_shape"])
    x = Conv(img_input, config, num_filter=64, kernel_size=3, padding="same", stride=2, name="conv_1")
    # 56 x 56 x 64
    if blocks[0] == 1:
        x = Conv(x, config, group=True, num_filter=64, kernel_size=3, padding="same", stride=1, name="conv_2_dw")
    else:
        x = Residual(x, config, num_block=blocks[0], num_out=64, kernel_size=(3, 3), stride=(1, 1), padding="same",
                             num_group=64, name="res_2")

    x = DResidual(x, config, num_out=64, kernel_size=(3, 3), stride=(2, 2), padding="same", num_group=128, name="dconv_23")
    # 28 x 28 x 64
    x = Residual(x, config, num_block=blocks[1], num_out=64, kernel_size=(3, 3), stride=(1, 1), padding="same", num_group=128,
                      name="res_3")
    x = DResidual(x, config, num_out=128, kernel_size=(3, 3), stride=(2, 2), padding="same", num_group=256, name="dconv_34")
    # 14 x 14 x 128
    x = Residual(x, config, num_block=blocks[2], num_out=128, kernel_size=(3, 3), stride=(1, 1), padding="same",
                      num_group=256, name="res_4")
    x = DResidual(x, config, num_out=128, kernel_size=(3, 3), stride=(2, 2), padding="same", num_group=512, name="dconv_45")
    # 7 x 7 x 128
    x = Residual(x, config, num_block=blocks[3], num_out=128, kernel_size=(3, 3), stride=(1, 1), padding="same",
                      num_group=256, name="res_5")
    x = Conv(x, config, num_filter=512, kernel_size=(1, 1), padding="valid", stride=(1, 1), name="conv_6sep")

    fc1 = get_fc1(x, config)

    embeds = keras.layers.Lambda(lambda x: K.l2_normalize(x))(fc1)

    if not is_train:
        model = models.Model(img_input, embeds, name=config["network"])
        return model

    logits = get_call_func(y_true, embeds, config)

    model = models.Model(img_input, logits, name=config["network"])
    # Load weights.
    return model, fc1
