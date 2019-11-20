import os
import argparse
import numpy as np
import math
import yaml

import tensorflow as tf
from tensorflow.python import keras as keras
from datetime import datetime

from libs.datatool import ImageData
from libs.utils import get_model_by_config, check_folders, load_bin, evaluate, get_port, run_embds

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='path to config file', default='./configs/fmobilefacenet_webface.yaml')
    parser.add_argument('--restore_weights', type=str, help='weights to restore train path. for example: ./output_dir/weight.h5', default='')
    return parser.parse_args()

class Trainer:
    def __init__(self, config):
        config["output_dir"] = os.path.join(config['outputs_dir'], datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
        check_folders([config["output_dir"]])
        config["log"] = os.path.join(config["output_dir"], 'log.txt')
        if config["dataset_size"] != None:
            config["step_per_epoch"] = math.ceil(config["dataset_size"]/config["batch_size"])
        with open(os.path.join(config["output_dir"], 'config.yaml'), 'w') as f:
            f.write(yaml.dump(config))
        self.config = config


    def build(self):
        config = self.config
#        cid = ImageData(img_size=config["image_size"], augment_flag=config['augment_flag'], augment_margin=config['augment_margin'])
#        train_dataset = cid.read_TFRecord(os.path.join("./data", config['train_data'])).shuffle(10000).repeat().batch(config["batch_size"])
#        train_iterator = train_dataset.make_one_shot_iterator()
#        self.train_images, self.train_labels = train_iterator.get_next()

#        print("image: ", self.train_images.get_shape())
#        print("labels ", self.train_labels.get_shape())
        image_data = ImageData()
        data_gen = image_data.generator(config)
        config["dataset_size"] = data_gen.samples
        config["class_num"] = data_gen.num_classes
        self.data_gen = data_gen
        def init_model():
            model = get_model_by_config(config, True)
            if os.path.exists(config["restore_weights"]):
                model.load_weights(config["restore_weights"])
            return model
                # logits = get_call_func(self.train_labels, self.model.output, config)

        if config["gpu_num"] <= 1:
            self.single_model = init_model()
            self.parallel_model = self.single_model
        else:
            self.single_model = init_model()
            print("Gpu num", config["gpu_num"])
            self.parallel_model = keras.utils.multi_gpu_model(self.single_model, gpus=config["gpu_num"])

        train_op = keras.optimizers.RMSprop(lr=0.001)
        from libs.loss import generate_loss_func
       # self.model.compile(self.train_op,loss=self.inference_loss)
        self.parallel_model.compile(train_op, loss=generate_loss_func(config), metrics=["acc"])

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
        config = self.config

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
        self.func = K.function([self.single_model.input], [self.single_model.output])
        class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                config = outter_class.config
                acc = []
                with open(config["log"], 'a') as f:
                    epoch_model_path = os.path.join(config["output_dir"], "tmp.h5")
                    outter_class.single_model.save(epoch_model_path)
                    f.write('step: %d\n' % counter)
                    for k, v in config["val_data"].items():
                        imgs, imgs_f, issame = load_bin(os.path.join("data", config["train_data"], v), config["image_size"])
                        embds = run_embds(outter_class.func, imgs, config["batch_size"]//config["gpu_num"])
                        embds_f = run_embds(outter_class.func, imgs_f, config["batch_size"]//config["gpu_num"])
                        embds = embds / np.linalg.norm(embds, axis=1, keepdims=True) + embds_f / np.linalg.norm(embds_f, axis=1, keepdims=True)
                        tpr, fpr, acc_mean, acc_std, tar, tar_std, far = evaluate(embds, issame, far_target=1e-3, distance_metric=0)
                        f.write('eval on %s: acc--%1.5f+-%1.5f, tar--%1.5f+-%1.5f@far=%1.5f\n' % (
                        k, acc_mean, acc_std, tar, tar_std, far))
                        print('eval on %s: acc--%1.5f+-%1.5f, tar--%1.5f+-%1.5f@far=%1.5f\n' % (
                        k, acc_mean, acc_std, tar, tar_std, far))
                        acc.append(acc_mean)
                    acc = np.mean(np.array(acc))
                    if acc > outter_class.best_acc:
#                        saver_best.save(sess, os.path.join(self.model_dir, 'best-m'), global_step=counter)
                        outter_class.best_acc = acc
                        outter_class.single_model.save(os.path.join(config["output_dir"], "{}_{}_model.h5".format(config["network"], config["train_data"])))
                    outter_class.single_model.load_weights(epoch_model_path)

            def on_batch_end(self, batch, logs=None):
                if batch % 1000 == 0:
                    config = outter_class.config
                    json_config = outter_class.single_model.to_json()
                    with open(os.path.join(config["output_dir"], "batch_model.json"), 'w') as json_file:
                        json_file.write(json_config)
                    # Save weights to disk
                    outter_class.single_model.save_weights(os.path.join(config["output_dir"], "batch_weights.h5"))




        workers = int(os.cpu_count() // 1.5)
        os.cpu_count()
        self.parallel_model.fit_generator(
            self.data_gen,
            epochs=config["epoch_num"],
            steps_per_epoch=config["step_per_epoch"],
            callbacks=[LossAndErrorPrintingCallback()],
            max_queue_size=workers*2,
            workers=workers,
            use_multiprocessing=True,
        )
        # construct the training image generator for data augmentation
        #self.parallel_model.fit(self.train_images, self.train_labels, batch_size=config["batch_size"],
        #               epochs=config["epoch_num"],
        #               steps_per_epoch=config["step_per_epoch"],
        #               #callbacks=[LossAndErrorPrintingCallback, tensorboard_callback])
        #               callbacks=[LossAndErrorPrintingCallback()])

if __name__ == '__main__':
    args = parse_args()

    config = yaml.load(open(args.config_path))
    weights_path = args.restore_weights
    config["restore_weights"] = weights_path

    trainer = Trainer(config)
    trainer.train()


