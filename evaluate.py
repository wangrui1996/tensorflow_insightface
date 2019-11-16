import io
import os
import yaml
import pickle
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

import cv2

from libs.utils import get_model_by_config,  load_bin, evaluate, run_embds

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
        model = get_model_by_config([112,112,3], config)
        model.load_weights(args.model_path)
        print(model.input_shape)
        func = K.function([model.input], [model.output])
        print('build done!')
        batch_size = config['batch_size']
        # batch_size = 32
        print('evaluating...')
        val_data = {}
        if args.val_data == '':
            val_data = config['val_data']
        else:
            val_data[os.path.basename(args.val_data)] = args.val_data
        for k, v in val_data.items():
            imgs, imgs_f, issame = load_bin(v, config['image_size'])
            print('img size is {}, imgs_f size is {}, issame is {}, and forward running...'.format(len(imgs), len(imgs_f), len(issame)))
            embds_arr = run_embds(func, imgs, batch_size)
            embds_f_arr = run_embds(func, imgs_f, batch_size)
            embds_arr = embds_arr/np.linalg.norm(embds_arr, axis=1, keepdims=True)+embds_f_arr/np.linalg.norm(embds_f_arr, axis=1, keepdims=True)
            print('get embds done!, starting to constract ...')
            tpr, fpr, acc_mean, acc_std, tar, tar_std, far = evaluate(embds_arr, issame, far_target=args.target_far, distance_metric=0)
            print('eval on %s: acc--%1.5f+-%1.5f, tar--%1.5f+-%1.5f@far=%1.5f' % (k, acc_mean, acc_std, tar, tar_std, far))
        print('done!')
    else:
        raise ValueError("Invalid value for --mode.")

