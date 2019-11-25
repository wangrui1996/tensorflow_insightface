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
    parser.add_argument('--retrain_weights', type=str, help='weights to restore train path. for example: ./output_dir/weight.h5', default='')
    return parser.parse_args()

class Trainer:
    def __init__(self, config):
        config["output_dir"] = os.path.join(config['outputs_dir'], config["network"], config["train_data"], config["loss_type"],datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
        check_folders([config["output_dir"]])
        config["log"] = os.path.join(config["output_dir"], 'log.txt')
        image_data = ImageData()
        print("starting to create data generator")
        data_gen = image_data.generator(config)
        print("Done ...")
        config["dataset_size"] = data_gen.samples
        config["class_num"] = data_gen.num_classes
        if config["dataset_size"] != None:
            config["step_per_epoch"] = math.ceil(config["dataset_size"]/config["batch_size"])
        with open(os.path.join(config["output_dir"], 'config.yaml'), 'w') as f:
            f.write(yaml.dump(config))
        self.config = config
        self.data_gen = data_gen


    def build(self):
        config = self.config

        def init_model():
            model, embeds = get_model_by_config(config, True)
            if os.path.exists(config["retrain_weights"]):
                model.load_weights(config["retrain_weights"])
            if os.path.exists(config["fine_weights"]):
                model.load_weights(config["fine_weights"])
            return model, embeds
                # logits = get_call_func(self.train_labels, self.model.output, config)

        if config["gpus"] <= 1:
            self.single_model, self.embeds = init_model()
            self.parallel_model = self.single_model
        else:
            self.single_model, self.embeds = init_model()
            print("Gpu num", config["gpus"])
            self.parallel_model = keras.utils.multi_gpu_model(self.single_model, gpus=config["gpus"])

        from libs.loss import LOSS_FUNC
        from libs.optimizers import OPTIMIZERS_WAY
        merics = ["acc"] if config["loss_type"] == "softmax" else []
        self.parallel_model.compile(OPTIMIZERS_WAY(config), loss=LOSS_FUNC(config), metrics=merics)

    def train(self):
        config = self.config

        self.build()
        from tensorflow.python.keras import backend as K
        func = K.function([self.single_model.input], [self.embeds])
        workers = int(os.cpu_count() // 1.5)
        self.parallel_model.fit_generator(
            self.data_gen,
            epochs=config["epoch_num"],
            steps_per_epoch=config["step_per_epoch"],
            callbacks=[TrainCallback(config, self.single_model, func)],
            max_queue_size=workers*2,
            workers=workers,
            use_multiprocessing=False,
        )

class TrainCallback(tf.keras.callbacks.Callback):

    def __init__(self, config, model, func):
        super(TrainCallback).__init__()
        self.config = config
        self.model = model
        self.func = func
        self.best_acc = 0


    def on_batch_end(self, batch, logs=None):
        config = self.config
        # set save model func
        if batch % config["snapshot"] == 0 and batch != 0:
            json_config = self.model.to_json()
            with open(os.path.join(config["output_dir"], "batch{}_model.json".format(batch)), 'w') as json_file:
                json_file.write(json_config)
            # Save weights to disk
            self.model.save_weights(os.path.join(config["output_dir"], "batch{}_weights.h5".format(batch)))

        # set test func
        elif batch % config["test_interval"] == 0 and batch != 0:
            acc = []
            with open(config["log"], 'a') as f:
                f.write('step: %d\n' % batch)
                for k, v in config["val_data"].items():
                    imgs, issame = load_bin(os.path.join("data", config["train_data"], v), config["image_size"])
                    embds = run_embds(self.func, imgs, config["batch_size"] // config["gpus"])
                    # embds_f = run_embds(outter_class.func, imgs_f, config["batch_size"]//config["gpus"])
                    # embds = embds / np.linalg.norm(embds, axis=1, keepdims=True) + embds_f / np.linalg.norm(embds_f, axis=1, keepdims=True)
                    tpr, fpr, acc_mean, acc_std, tar, tar_std, far = evaluate(embds, issame, far_target=1e-3,
                                                                                      distance_metric=0)
                    f.write('eval on %s: acc--%1.5f+-%1.5f, tar--%1.5f+-%1.5f@far=%1.5f\n' % (
                        k, acc_mean, acc_std, tar, tar_std, far))
                    print('eval on %s: acc--%1.5f+-%1.5f, tar--%1.5f+-%1.5f@far=%1.5f\n' % (
                        k, acc_mean, acc_std, tar, tar_std, far))
                    acc.append(acc_mean)
                acc = np.mean(np.array(acc))
                if acc > self.best_acc:
                    # saver_best.save(sess, os.path.join(self.model_dir, 'best-m'), global_step=counter)
                    self.best_acc = acc
                    self.model.save(os.path.join(config["output_dir"], "best_model.h5"))

if __name__ == '__main__':
    args = parse_args()

    config = yaml.load(open(args.config_path), Loader=yaml.FullLoader)
    weights_path = args.retrain_weights
    config["retrain_weights"] = weights_path

    trainer = Trainer(config)
    trainer.train()


