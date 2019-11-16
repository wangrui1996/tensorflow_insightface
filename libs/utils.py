import tensorflow as tf
from tensorflow.python.keras import layers
import libs.models.resnet_v2 as resnet
import libs.models.fmobilefacenet as fmobilefacenet

import os
import cv2
import io

import math
import numpy as np

from sklearn.model_selection import KFold
from scipy import interpolate

def get_model_by_config(config):
    if config["network"] == "fmobilefacenet":
        model = fmobilefacenet.get_network(config)
    else:
        print("error")
        exit(0)
    return model




def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric == 0:
        # Euclidian distance
        embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, distance_metric=0, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    dist = distance(embeddings1, embeddings2, distance_metric)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_tar_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    tar = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return tar, far


def calculate_tar(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0,
                  subtract_mean=False):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tar = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_tar_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        tar[fold_idx], far[fold_idx] = calculate_tar_far(threshold, dist[test_set], actual_issame[test_set])

    tar_mean = np.mean(tar)
    far_mean = np.mean(far)
    tar_std = np.std(tar)
    return tar_mean, tar_std, far_mean




def check_folders(paths):
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def tensor_description(var):
    """Returns a compact and informative string about a tensor.
    Args:
        var: A tensor variable.
    Returns:
        a string with type and size, e.g.: (float32 1x8x8x1024).
    """
    description = '(' + str(var.dtype.name) + ' '
    sizes = var.get_shape()
    for i, size in enumerate(sizes):
        description += str(size)
        if i < len(sizes) - 1:
            description += 'x'
    description += ')'
    return description


def analyze_vars(variables, path):
    """Prints the names and shapes of the variables.
    Args:
        variables: list of variables, for example tf.global_variables().
        print_info: Optional, if true print variables and their shape.
    Returns:
        (total size of the variables, total bytes of the variables)
    """
    f = open(path, 'w')
    f.write('---------\n')
    f.write('Variables: name (type shape) [size]\n')
    f.write('---------\n')
    total_size = 0
    total_bytes = 0
    for var in variables:
        # if var.num_elements() is None or [] assume size 0.
        var_size = var.get_shape().num_elements() or 0
        var_bytes = var_size * var.dtype.size
        total_size += var_size
        total_bytes += var_bytes
        f.write(var.name+' '+tensor_description(var)+' '+'[%d, bytes: %d]\n' % (var_size, var_bytes))
    f.write('Total size of variables: %d\n' % total_size)
    f.write('Total bytes of variables: %d\n' % total_bytes)
    return total_size, total_bytes

def load_bin(path, image_size):
    print('reading %s\r' % path)
    import pickle
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    num = len(bins)
    images = np.zeros(shape=[num, image_size, image_size, 3], dtype=np.float32)
    images_f = np.zeros(shape=[num, image_size, image_size, 3], dtype=np.float32)
    # m = config['augment_margin']
    # s = int(m/2)
    cnt = 0
    for bin in bins:
        img = np.fromstring(io.BytesIO(bin).read(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        # img = img[s:s+image_size, s:s+image_size, :]
        img_f = np.fliplr(img)
        img = img/127.5-1.0
        img_f = img_f/127.5-1.0
        images[cnt] = img
        images_f[cnt] = img_f
        cnt += 1
    print('done!\r')
    return (images, images_f, issame_list)



def evaluate(embeddings, actual_issame, far_target=1e-3, distance_metric=0, nrof_folds=10):
    thresholds = np.arange(0, 4, 0.01)
    if distance_metric == 1:
        thresholdes = np.arange(0, 1, 0.0025)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2, np.asarray(actual_issame), distance_metric=distance_metric, nrof_folds=nrof_folds)
    tar, tar_std, far = calculate_tar(thresholds, embeddings1, embeddings2, np.asarray(actual_issame), far_target=far_target, distance_metric=distance_metric, nrof_folds=nrof_folds)
    acc_mean = np.mean(accuracy)
    acc_std = np.std(accuracy)
    return tpr, fpr, acc_mean, acc_std, tar, tar_std, far


def run_embds(func, images, batch_size):
    batch_num = len(images)//batch_size
    left = len(images)%batch_size
    embds = []
    for i in range(batch_num):
        image_batch = images[i*batch_size: (i+1)*batch_size]
        cur_embd = func([image_batch])[0]
        embds += list(cur_embd)
        print('%d/%d' % (i, batch_num), end='\r')
    if left > 0:
        image_batch = images[-left:]
        cur_embd = func([image_batch])[0]
        embds += list(cur_embd)
    print('get features done!\r')
    return np.array(embds)


def get_port(num=1):
    def tryPort(port):
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = False
        try:
            sock.bind(("0.0.0.0", port))
            result = True
        except:
            pass
        sock.close()
        return result

    port = []
    for i in range(1024, 65535):
        if tryPort(i):
            port.append(i)
            if len(port) == num:
                return port
