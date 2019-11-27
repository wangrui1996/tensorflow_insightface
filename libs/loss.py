import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
from tensorflow.python import keras
from tensorflow.python.ops import array_ops

def _get_available_devices():
  return [x.name for x in K.get_session().list_devices()]


def _normalize_device_name(name):
  name = '/' + name.lower().split('device:')[1]
  return name

def generate_loss_func(config):
    from tensorflow.python.util.tf_export import keras_export
    if config["gpus"] <= 1:
        @keras_export("keras.losses.custom_loss")
        def _loss(y_true, y_pred, from_logits=False,
                             label_smoothing=0):
            return loss_inference(y_true, y_pred, from_logits,
                             label_smoothing,config)
        return _loss
    else:
        @keras_export("keras.losses.custom_loss")
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


def loss_inference(y_true, y_pred, from_logits=False, label_smoothing=0, config=None):
    if config['loss_type'] == "softmax":
        pass
    elif config['loss_type'] == 'margin':
        y_pred = margin_softmax(y_pred, y_true, config)
    else:
        raise ValueError('Invalid loss type.')
#    from tensorflow.python.ops import math_ops
#    from tensorflow.python.framework import smart_cond
#    y_pred = ops.convert_to_tensor(y_pred)
#    y_true = math_ops.cast(y_true, y_pred.dtype)
#    label_smoothing = ops.convert_to_tensor(0, dtype=K.floatx())

#    def _smooth_labels():
#        num_classes = math_ops.cast(array_ops.shape(y_true)[1], y_pred.dtype)
#        return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

#    y_true = smart_cond.smart_cond(label_smoothing,
#                                       _smooth_labels, lambda: y_true)
    main_loss = K.categorical_crossentropy(y_true, y_pred, True)
#    main_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
    # inference_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
    if config['ce_loss']:
        body = y_pred
        body = K.log(body)
        _label = -y_true
        #_label = tf.one_hot(y_true_int, depth=config["class_num"], on_value=-1.0, off_value=0.0)
        body = body * _label
        ce_loss = K.sum(body) / config["batch_size"]
        train_loss = main_loss + ce_loss
    else:
        train_loss = main_loss
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
        print("output shape", output.shape)
        print("finished ..")
    return output


LOSS_FUNC = generate_loss_func
