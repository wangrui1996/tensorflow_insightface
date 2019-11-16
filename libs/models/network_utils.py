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
        x = layers.Dense(config["embd_size"], name="pre_fc1")(x)
        x = layers.BatchNormalization(momentum=config["bn_mom"], epsilon=2e-5, name='fc1')(x)
    return x