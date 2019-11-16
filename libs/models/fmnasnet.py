import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.backend import backend

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


def get_network(config, m=1.0):

    img_input = layers.Input(shape=config["input_shape"])

    interverted_residual_setting = [
        # t, c,  n, s, k
        [3, int(24*m),  3, 2, 3, "stage2_"],  # -> 56x56
        [3, int(40*m),  3, 2, 5, "stage3_"],  # -> 28x28
        [6, int(80*m),  3, 2, 5, "stage4_1_"],  # -> 14x14
        [6, int(96*m),  2, 1, 3, "stage4_2_"],  # -> 14x14
        [6, int(192*m), 4, 2, 5, "stage5_1_"], # -> 7x7
        [6, int(320*m), 1, 1, 3, "stage5_2_"], # -> 7x7
    ]

    last_channels = int(1024*m)






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

    model = models.Model(img_input, fc1, name=config["network"])
    # Load weights.
    if config["fine_weights"] != "None":
        model.load_weights(config["fine_weights"])
    return model
