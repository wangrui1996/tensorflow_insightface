import io
import os
import yaml
import pickle
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

import cv2

from libs.utils import get_model_by_config, test

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='build', help='model mode: build')
    parser.add_argument('--config_path', type=str, default='./configs/config_ms1m_100.yaml', help='config path, used when mode is build')
    parser.add_argument('--model_path', type=str, default='./model_dir/weights.h5', help='model path')
    parser.add_argument('--val_data', type=str, default='', help='val data, a dict with key as data name, value as data path')
    parser.add_argument('--target_far', type=float, default=1e-3, help='target far when calculate tar')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    if args.mode == 'build':
        print('building model ...')
        config = yaml.load(open(args.config_path))
        model = get_model_by_config(config, is_train=False)
        model.load_weights(args.model_path, True)
        print(model.input_shape)
        func = K.function([model.input], [model.output])
        print('build done!')
        batch_size = config['batch_size']
        # batch_size = 32
        print('evaluating...')
        val_data = {}
        if args.val_data == '':
            val_data = { k : os.path.join('data', config["train_data"], v) for k,v in config['val_data'].items()}
        else:
            val_data[os.path.basename(args.val_data)] = args.val_data
        for k, v in val_data.items():
            acc_mean, acc_std, best_threshold, dist_min, dist_max = test(v, config, func)
            print('eval %s. Accuracy-Flip: %1.5f+-%1.5f' % (k, acc_mean, acc_std))
            print('best threshold %1.5f. distance range (%1.5f-%1.5f)' % (best_threshold, dist_min, dist_max))
    else:
        raise ValueError("Invalid value for --mode.")

