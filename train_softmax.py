import os
import argparse
import numpy as np
import math
import yaml

import tensorflow as tf
from tensorflow.python import keras as keras
from datetime import datetime

from libs.loss import get_call_func
from libs.datatool import ImageData
from libs.utils import get_model_by_config, check_folders, load_bin, evaluate, get_port, run_embds

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='path to config file', default='./configs/fmobilefacenet_webface.yaml')
    return parser.parse_args()

class Trainer:
    def __init__(self, config):
        self.config = config
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        self.output_dir = os.path.join(config['output_dir'], subdir)
        self.model_dir = os.path.join(self.output_dir, 'models')
        self.log_dir = os.path.join(self.output_dir, 'log')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.debug_dir = os.path.join(self.output_dir, 'debug')
        check_folders([self.output_dir, self.model_dir, self.log_dir, self.checkpoint_dir, self.debug_dir])
        self.val_log = os.path.join(self.output_dir, 'val_log.txt')

        self.batch_size = config['batch_size']
        self.gpu_num = config['gpu_num']
        if self.batch_size % self.gpu_num != 0:
            raise ValueError('batch_size must be a multiple of gpu_num')
        self.image_size = config['image_size']
        self.epoch_num = config['epoch_num']
        self.step_per_epoch = config['step_per_epoch']
        self.val_freq = config['val_freq']
        self.val_data = config['val_data']

        with open(os.path.join(self.output_dir, 'config.yaml'), 'w') as f:
            f.write(yaml.dump(self.config))


    def build(self):
        cid = ImageData(img_size=self.image_size, augment_flag=self.config['augment_flag'], augment_margin=self.config['augment_margin'])
#        dataset_size = cid.get_dataset_size(self.config["train_data"])
        if self.config["dataset_size"] != None:
            self.step_per_epoch = math.ceil(self.config["dataset_size"]/self.config["batch_size"])
        train_dataset = cid.read_TFRecord(self.config['train_data']).shuffle(10000).repeat().batch(self.config["batch_size"])
        train_iterator = train_dataset.make_one_shot_iterator()
        self.train_images, self.train_labels = train_iterator.get_next()
        print("image: ", self.train_images.get_shape())
        print("labels ", self.train_labels.get_shape())

        self.model = get_model_by_config(config)
        self.inference_loss, logits = get_call_func(self.train_labels, self.model.output, config)

        if self.gpu_num > 1:
            self.model = keras.utils.multi_gpu_model(self.model, gpus=self.gpu_num)
        self.embds = self.model.output

        def metrics_func(_, __):
            pred = tf.argmax(logits, axis=-1, output_type=tf.int64)
            train_acc = tf.reduce_mean(tf.cast(tf.equal(pred, self.train_labels), tf.float32))
            return train_acc

#            self.wd_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
#            self.train_loss = self.inference_loss+self.wd_loss

        self.train_op = keras.optimizers.RMSprop(lr=0.1)
       # self.model.compile(self.train_op,loss=self.inference_loss)
        self.model.compile(self.train_op, loss=self.inference_loss, metrics=[metrics_func])

    def set_tf_config(self, num_workers):
        import json
        tf_config = json.dumps({
            'cluster': {
                'worker': []
            },
            'task': {'type': 'worker', 'index': 0}
        })
        tf_config = json.loads(tf_config)
        for port in get_port(num_workers):
            tf_config["cluster"]["worker"].append("localhost:{}".format(port))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)

    def train(self):

#        NUM_WORKERS = math.ceil(self.config["batch_size"] / self.config["gpu_capacity"])
#        self.set_tf_config(NUM_WORKERS)
#        GLOBAL_BATCH_SIZE = config["batch_size"] * NUM_WORKERS
#        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
#        with strategy.scope():
#            self.build(GLOBAL_BATCH_SIZE)
#            self.config["batch_size"] = GLOBAL_BATCH_SIZE
        self.build()
        self.best_acc = 0
        counter = 0
        outter_class = self
        from tensorflow.python.keras import backend as K
        self.func = K.function([self.model.input], [self.model.output])
        class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                acc = []
                with open(outter_class.val_log, 'a') as f:
                    epoch_model_path = os.path.join(outter_class.config["output_dir"], "model.h5")
                    outter_class.model.save(epoch_model_path)
                    f.write('step: %d\n' % counter)
                    for k, v in outter_class.val_data.items():
                        imgs, imgs_f, issame = load_bin(v, outter_class.image_size)
                        embds = run_embds(outter_class.func, imgs, outter_class.batch_size)
                        embds_f = run_embds(outter_class.func, imgs_f, outter_class.batch_size)
                        embds = embds / np.linalg.norm(embds, axis=1, keepdims=True) + embds_f / np.linalg.norm(embds_f,
                                                                                                                axis=1,
                                                                                                                keepdims=True)
                        tpr, fpr, acc_mean, acc_std, tar, tar_std, far = evaluate(embds, issame, far_target=1e-3,
                                                                                  distance_metric=0)
                        f.write('eval on %s: acc--%1.5f+-%1.5f, tar--%1.5f+-%1.5f@far=%1.5f\n' % (
                        k, acc_mean, acc_std, tar, tar_std, far))
                        print('eval on %s: acc--%1.5f+-%1.5f, tar--%1.5f+-%1.5f@far=%1.5f\n' % (
                        k, acc_mean, acc_std, tar, tar_std, far))
                        acc.append(acc_mean)
                    acc = np.mean(np.array(acc))
                    if acc > outter_class.best_acc:
#                        saver_best.save(sess, os.path.join(self.model_dir, 'best-m'), global_step=counter)
                        outter_class.best_acc = acc
                    outter_class.model.load_weights(epoch_model_path)

            def on_batch_end(self, batch, logs=None):
                if batch % 1000 == 0:
                    json_config = outter_class.model.to_json()
                    with open(outter_class.config["model_config"], 'w') as json_file:
                        json_file.write(json_config)

                    # Save weights to disk
                    outter_class.model.save_weights(outter_class.config["model_weights"])

        model_weights = self.config["model_weights"]
        if os.path.exists(model_weights):
            self.model.load_weights(model_weights)

        self.model.fit(self.train_images, self.train_labels, batch_size=self.config["batch_size"], epochs=100000,
                       steps_per_epoch=self.step_per_epoch,
 #                      steps_per_epoch=2,
                       callbacks=[LossAndErrorPrintingCallback()])

if __name__ == '__main__':
    args = parse_args()
    config = yaml.load(open(args.config_path))
    trainer = Trainer(config)
    trainer.train()


