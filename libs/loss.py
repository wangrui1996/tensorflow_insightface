import tensorflow as tf
import math
import numpy as np
from tensorflow.python.keras import layers
import tensorflow.keras.backend as K
from tensorflow.python import keras
from tensorflow.python.keras import regularizers

def generate_loss_func(config):
    def loss_func(y_true, y_pred):




        if config['loss_type'] == "softmax":
            logits = keras.layers.Dense(config['class_num'], use_bias=config["fc7_use_bias"], name="fc7")(y_pred)
        elif config['loss_type'] == 'margin':
            y_true = None
            logits = margin_softmax(y_pred, y_true, config)
        else:
            raise ValueError('Invalid loss type.')
        from tensorflow.python.keras import layers
        inference_loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y_true)
        #inference_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
        if config['ce_loss']:
            body = layers.Softmax()(logits)
            body = K.log(body)
            _label = tf.one_hot(y_true, depth=config["class_num"], on_value=-1.0, off_value=0.0)
            body = body * _label
            ce_loss = K.sum(body) / config["batch_size"]
            train_loss = inference_loss + ce_loss + tf.compat.v1.losses.get_regularization_loss()
        else:
            train_loss = inference_loss + tf.compat.v1.losses.get_regularization_loss()
        return train_loss
    return loss_func

def margin_softmax(embedding, y_true, config):
    s = config["loss_s"]
    def mul_s(x):
        return x*s
    #nembedding = keras.layers.Lambda(lambda x:  embedding*s)(embedding)
    nembedding = keras.layers.Lambda(mul_s)(embedding)
    fc7 = layers.Dense(units=config["class_num"], use_bias=False, kernel_regularizer=K.l2_normalize, name="cos0")(nembedding)

    if config["loss_m1"] == 1.0 and config["loss_m2"] == 0.0:
        s_m = s*config["loss_m3"]
        gt_one_hot = keras.layers.Lambda(
            lambda label: tf.one_hot(label, depth=config["class_num"], on_value=s_m, off_value=0.0))(y_true)
        def sub_label(x):
            return x-gt_one_hot
        output = keras.layers.Lambda(sub_label)(fc7)
    else:
        zy = fc7
        def div_s(x):
            return x/s
        cos_t = keras.layers.Lambda(div_s)(zy)
        t = keras.layers.Lambda(lambda x: tf.math.acos(x))(cos_t)
        if config["loss_m1"] != 1.0:
            def mul_m1(x):
                return x*config["loss_m1"]
            t = keras.layers.Lambda(mul_m1)(t)
        if config["loss_m2"] > 0.0:
            def add_m2(x):
                return x+config["loss_m2"]
            t = keras.layers.Lambda(add_m2)(t)
        body = keras.layers.Lambda(lambda x: K.cos(x))(t)
        if config["loss_m3"] > 0.0:
            def sub_m3(x):
                return x - config["loss_m3"]
            body = keras.layers.Lambda(sub_m3)(body)

        new_zy = keras.layers.Lambda(mul_s)(body)
        def bool_one_hot_func(ip):
            return K.one_hot(ip[0], ip[1])
            #return tf.one_hot(ip[0], depth=ip[1])

        #bool_one_hot = keras.layers.Lambda(lambda x,y: K.one_hot(x,y))(y_true, config["class_num"])
        bool_one_hot = np.ones((config["batch_size"], config["class_num"]))
        #bool_one_hot = K.one_hot(y_true, config["class_num"])
        #bool_one_hot = keras.layers.Lambda(bool_one_hot_func)([y_true, config["class_num"]])
        def where_func(ip):
            x = tf.cast(ip[0], dtype=tf.bool)
            return tf.where(x, ip[1], ip[2])
        output = keras.layers.Lambda(where_func)([bool_one_hot, new_zy, fc7])
        print("finished ..")
    return output



