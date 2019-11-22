import tensorflow as tf
import math
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.keras import layers
import tensorflow.keras.backend as K
from tensorflow.python import keras
from tensorflow.python.ops import array_ops
from tensorflow.python.keras import regularizers

def _get_available_devices():
  return [x.name for x in K.get_session().list_devices()]


def _normalize_device_name(name):
  name = '/' + name.lower().split('device:')[1]
  return name

def generate_loss_func(config):
    if config["gpus"] <= 1:
        def _loss(y_true, y_pred):
            return loss_inference(y_true, y_pred, config)
        return _loss
    else:
        def _multi_gpu_loss(y_true, y_pred):
            from tensorflow.python.keras.layers.core import Lambda
            gpus = config["gpus"]
            if isinstance(gpus, (list, tuple)):
                if len(gpus) <= 1:
                    raise ValueError('For multi-gpu usage to be effective, '
                                     'call `multi_gpu_model` with `len(gpus) >= 2`. '
                                     'Received: `gpus=%s`' % gpus)
                num_gpus = len(gpus)
                target_gpu_ids = gpus
            else:
                if gpus <= 1:
                    raise ValueError('For multi-gpu usage to be effective, '
                                     'call `multi_gpu_model` with `gpus >= 2`. '
                                     'Received: `gpus=%s`' % gpus)
                num_gpus = gpus
                target_gpu_ids = range(num_gpus)

            target_devices = ['/cpu:0'] + ['/gpu:%d' % i for i in target_gpu_ids]
            available_devices = _get_available_devices()
            available_devices = [
                _normalize_device_name(name) for name in available_devices
            ]
            for device in target_devices:
                if device not in available_devices:
                    raise ValueError('To call `multi_gpu_model` with `gpus=%s`, '
                                     'we expect the following devices to be available: %s. '
                                     'However this machine only has: %s. '
                                     'Try reducing `gpus`.' % (gpus, target_devices,
                                                               available_devices))

            def get_slice(data, i, parts):
                """Slice an array into `parts` slices and return slice `i`.

                Arguments:
                  data: array to slice.
                  i: index of slice to return.
                  parts: number of slices to make.

                Returns:
                  Slice `i` of `data`.
                """
                shape = array_ops.shape(data)
                batch_size = shape[:1]
                input_shape = shape[1:]
                step = batch_size // parts
                if i == parts - 1:
                    size = batch_size - step * i
                else:
                    size = step
                size = array_ops.concat([size, input_shape], axis=0)
                stride = array_ops.concat([step, input_shape * 0], axis=0)
                start = stride * i
                return array_ops.slice(data, start, size)
                # Place a copy of the model on each GPU,
                # each getting a slice of the inputs.
            outputs = []
            for i, gpu_id in enumerate(target_gpu_ids):
                with ops.device('/gpu:%d' % gpu_id):
                    with K.name_scope('loss_replica_%d' % gpu_id):
                        input = []
                        # Retrieve a slice of the input.
                        pred_shape = tuple(y_pred.shape.as_list())[1:]
                        pred_slice_i = Lambda(
                            get_slice,
                            output_shape=pred_shape,
                            arguments={
                                'i': i,
                                'parts': num_gpus
                            })(y_pred)
                        label_shape = tuple(y_true.shape.as_list())[1:]
                        label_slice_i = Lambda(
                            get_slice,
                            output_shape=label_shape,
                            arguments={
                                'i': i,
                                'parts': num_gpus
                            })(y_true)

                        # Apply model on slice
                        # (creating a model replica on the target device).
                        output = loss_inference(y_true=label_slice_i, y_pred=pred_slice_i, config=config)

                        # Save the outputs for merging back together later.
                        outputs.append(output)
            with ops.device('/cpu:0'):
                for id, ouput in enumerate(outputs):
                    if id == 0:
                        merged = output
                    else:
                        merged = merged + output
            return merged

        return _multi_gpu_loss


def loss_inference(y_true, y_pred, config):
    if config['loss_type'] == "softmax":
        logits = keras.layers.Dense(config['class_num'], use_bias=config["fc7_use_bias"], name="fc7")(y_pred)
    elif config['loss_type'] == 'margin':
        logits = margin_softmax(y_pred, y_true, config)
    else:
        raise ValueError('Invalid loss type.')
    from tensorflow.python.keras import layers
    main_loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=y_true)
    # inference_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
    if config['ce_loss']:
        batch_size = tuple(y_pred.shape.as_list())[0]
        body = layers.Softmax()(logits)
        body = K.log(body)
        _label = tf.one_hot(y_true, depth=config["class_num"], on_value=-1.0, off_value=0.0)
        body = body * _label
        ce_loss = K.sum(body) / batch_size
        train_loss = main_loss + ce_loss
    else:
        train_loss = main_loss
    train_loss = train_loss + tf.compat.v1.losses.get_regularization_loss()
    return train_loss


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

        bool_one_hot = K.cast(K.reshape(y_true, (-1, config["class_num"])), dtype=tf.bool)

        output = tf.where(bool_one_hot, new_zy, fc7)
        print("outpu shape", output.shape)
        print("finished ..")
    return output


LOSS_FUNC = generate_loss_func