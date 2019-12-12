import os
import argparse
import numpy as np
import math
import yaml

import tensorflow as tf
from tensorflow.python import keras as keras
from datetime import datetime

from libs.datatool import ImageData
from libs.utils import get_model_by_config, check_folders, load_bin, evaluate, get_port, run_embds, test

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
                model.load_weights(config["retrain_weights"], True)
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
        if os.path.exists(config["fine_weights"]):
            self.parallel_model.load_weights(config["fine_weights"], True)

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
        self.counter = 0

    def on_batch_end(self, batch, logs=None):
        counter = self.counter + 1
        config = self.config
        # set save model func
        if counter % config["snapshot"] == 0:
            json_config = self.model.to_json()
            with open(os.path.join(config["output_dir"], "step{}_model.json".format(counter)), 'w') as json_file:
                json_file.write(json_config)
            # Save weights to disk
            self.model.save(os.path.join(config["output_dir"], "step{}_weights.h5".format(counter)))

        # set test func
        if counter % config["test_interval"] == 0 or counter == 1:
            acc = []
            with open(config["log"], 'a') as f:
                f.write('step: %d\n' % counter)
                for k, v in config["val_data"].items():
                    acc_mean, acc_std, best_threshold, dist_min, dist_max = test(os.path.join("data", config["train_data"], v), config, self.func)
                    print('eval %s. Accuracy-Flip: %1.5f+-%1.5f' % (k, acc_mean, acc_std))
                    print('best threshold %1.5f. distance range (%1.5f-%1.5f)' % (best_threshold, dist_min, dist_max))
                    f.write('eval %s. Accuracy-Flip: %1.5f+-%1.5f' % (k, acc_mean, acc_std))
                    f.write('best threshold %1.5f. distance range (%1.5f-%1.5f)' % (best_threshold, dist_min, dist_max))
                    acc.append(acc_mean)
                acc = np.mean(np.array(acc))
                if acc > self.best_acc:
                    # saver_best.save(sess, os.path.join(self.model_dir, 'best-m'), global_step=counter)
                    self.best_acc = acc
                    self.model.save(os.path.join(config["output_dir"], "best_model.h5"))
        self.counter = counter

if __name__ == '__main__':
    args = parse_args()

    config = yaml.load(open(args.config_path), Loader=yaml.FullLoader)
    weights_path = args.retrain_weights
    config["retrain_weights"] = weights_path

    trainer = Trainer(config)
    trainer.train()


