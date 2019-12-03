import yaml
import argparse

from tensorflow.python.framework import ops
from tensorflow.python.keras.utils import multi_gpu_model
from libs.utils import get_model_by_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='path to config file', default='./configs/fmobilefacenet_webface.yaml')
    parser.add_argument('--gpus_weight', type=str, help='weights to restore train path. for example: ./output_dir/weight.h5', default='')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    config = yaml.load(open(args.config_path), Loader=yaml.FullLoader)
    weights_path = args.gpus_weight
    config["retrain_weights"] = weights_path
    with ops.device("/cpu:0"):
        model, _ = get_model_by_config(config, True)
    para_model = multi_gpu_model(model, config["gpus"])
    para_model.load_weights(weights_path)
    model.save("/home/rui/single.h5")

