from tensorflow.python import keras as keras

def get_optimizers_from_config(config):
    optimizers_way = config["optimizers"]
    base_lr = config["base_lr"]
    if optimizers_way == "sgd":
        return keras.optimizers.SGD(lr=base_lr, )

    if optimizers_way == "adam":
        return keras.optimizers.Adam(lr=base_lr)

    if optimizers_way == "rmsp":
        return keras.optimizers.RMSprop(lr=base_lr)

    print("error get optimizers type: {}".format(optimizers_way))
    exit(0)

OPTIMIZERS_WAY=get_optimizers_from_config