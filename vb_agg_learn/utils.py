from __future__ import division
import os
import pickle
from contextlib import contextmanager
from functools import partial

import numpy as np
import tensorflow as tf
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import pdist, squareform

# Malaria strategic partition (split into 10 parts and combine)
def partition_maker(feats_y, bag_pop, estop_size, val_size, test_size, rs):
    # Only works for 10% partitions.
    y = np.divide(feats_y, bag_pop)
    for size in [estop_size, val_size, test_size]:
        if size % 0.1 != 0:
            raise ValueError('Code only works for 10 percent partitions')
    sorted_ind = np.argsort(y)
    partitions = [ [] for i in range(0, 10) ]
    l = 0
    for ind in sorted_ind:
        partitions[l].append(ind)
        if l == 9:
            l = 0 
        else:
            l = l + 1
    rs.shuffle(partitions) #make it random
    # Count how many parts (integer division)
    train_part = int((1.0 - estop_size - val_size - test_size) / 0.1)
    estop_part = int(estop_size / 0.1)
    val_part = int(val_size / 0.1)
    test_part = int(test_size / 0.1)
    tr_ind = np.concatenate(partitions[:train_part])
    es_ind = np.concatenate(partitions[train_part:train_part+estop_part])
    val_ind = np.concatenate(partitions[train_part+estop_part:train_part+estop_part+val_part])
    te_ind = np.concatenate(partitions[train_part+estop_part+val_part:])
    return tr_ind, es_ind, val_ind, te_ind

    
def true_dimension(dataset): # computes true dimension, ignoring obs_y, true_y and population
    if dataset.indiv and dataset.true_indiv and dataset.pop:
        true_dim = dataset.dim - 3
    elif dataset.indiv and dataset.pop:
        true_dim = dataset.dim - 2
    elif dataset.indiv or dataset.pop:
        true_dim = dataset.dim - 1
    else:
        true_dim = dataset.dim
    return true_dim

# Toy Norm 
def elevators_norm(labels, min_l=0.12, max_l=0.78):
    return (labels - min_l) / (max_l - min_l)

def swiss_norm(labels, min_l=4.7124, max_l=14.137): 
    return (labels - min_l) / (max_l - min_l)

def PSD(A, reg=1e-4, tol=0.0): # PSD
    evals, eV = np.linalg.eig(A)
    evals = np.real(evals) #due to numerical error
    eV = np.real(eV)
    if not np.all(evals > tol): #small tolerance allowed
        if isinstance(reg, float) and reg > 0.0:
            ev_small = np.sort(evals[evals > 0])[0]
            evals[evals <= 0] = min(reg, ev_small) #if reg too large
        else:
            raise ValueError('float {} is not positive float'.format(reg))
    #print(evals)
    psd_A = eV.dot(np.diag(evals)).dot(eV.T) # reconstruction
    return psd_A
'''
def symmetric_matrix_square_root(mat, eps=1e-10):
    """Compute square root of a symmetric matrix.
    Note that this is different from an elementwise square root. We want to
    compute M' where M' = sqrt(mat) such that M' * M' = mat.
    Also note that this method **only** works for symmetric matrices.
    Args:
      mat: Matrix to take the square root of.
      eps: Small epsilon such that any element less than eps will not be square
        rooted to guard against numerical instability.
    Returns:
      Matrix square root of mat.
    """
    # Unlike numpy, tensorflow's return order is (s, u, v)
    s, u, v = tf.svd(mat)
    #s = tf.Print(s, [s], summarize=10000, message='s')
    # sqrt is unstable around 0, just use 0 in such case
    #si = tf.where(tf.less(s, eps), s, tf.sqrt(s))
    # Note that the v returned by Tensorflow is v = V
    # (when referencing the equation A = U S V^T)
    # This is unlike Numpy which returns v = V^T
    return tf.matmul(
        tf.matmul(u, tf.diag(tf.divide(1.0, s))), tf.transpose(v)) 
'''
def extract(features):
    if features.pop and features.indiv:
        data = features[:,:-2]
    elif features.pop:
        data = features[:,:-1]
    elif features.indiv:
        data = features[:,:-1]
    else:
        data = features
    return data

def standardise(labels):
    min_l = np.min(labels)
    max_l = np.max(labels)
    return (labels - min_l) / (max_l - min_l)

def increase_dim(X, dim):
    data_size, latent_dim = X.shape
    zeros_dim = dim - latent_dim
    zeros_data = np.zeros((data_size, zeros_dim))
    X_dim = np.hstack((X, zeros_data))
    return X_dim

def check_positive(value):
    float_v = float(value)
    if float_v < 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive float value" % float_v)
    return float_v

# Loop for batches 
def loop_batches(feats, max_pts, max_bags=np.inf, shuffle=False, stack=False, rs=None):
    #print('max bags:', max_bags)
    #print('max pts:', max_pts)
    '''
    Loop over feats, yielding subsets with no more than max_pts total points
    in them and no more than max_bags bags.
    '''
    if shuffle:
        #feats = feats[rs.permutation(len(feats))]  # doesn't copy data
        feats = feats[rs.permutation(len(feats))]
    rest = feats
    #print(rest[0][0])
    while len(rest):
        pts_i = np.cumsum(rest.n_pts).searchsorted(max_pts)  
        how_many = min(pts_i, max_bags)
        if how_many == 0:
            raise ValueError("Bag of size {} doesn't work with max_pts {}"
                             .format(rest.n_pts[0], max_pts))
        this = rest[:how_many]
        rest = rest[how_many:]
        if stack:
            this.make_stacked()
            # Provides a generator object instead of output and when used it will consume up.... 
        yield this

def load_split_elevators(es_plit=0.1, val_split=0.1, test_split=0.2, split_seed=23):
    assert es_plit + val_split + test_split < 1.0
    root_directory1 = 'data_path1'
    root_directory2 = 'data_path2'
    data_path1 = os.path.join(root_directory1, 'elevators.pkl')
    data_path2 = os.path.join(root_directory2, 'elevators.pkl')
    if os.path.isfile(data_path1):
        data = pickle.load(open( data_path1, "rb" ))
    elif os.path.isfile(data_path2):
        data = pickle.load(open( data_path2, "rb" ))
    else:
        raise ValueError('{} {} does not exist, check paths'.format(data_path1, data_path2))
    
    rs = check_random_state(split_seed)
    rs.shuffle(data)

    size = len(data)
    test_s = int(test_split * size)
    val_s = int(val_split * size)
    estop_s = int(es_plit * size)
    train_s = size - test_s - val_s - estop_s

    train = data[:train_s, :]
    val = data[train_s:(train_s + val_s), :]
    estop = data[(train_s + val_s):(train_s + val_s + estop_s), :]
    test = data[(train_s + val_s + estop_s):, :]
    return train, estop, val, test

# safe log 
def safe_log(y):
    y[y <= 0] = 1.0
    y = np.log(y)
    return y

# Load malaria + malaria with missing labels/covariates
def load_miss_malaria():
    root_directory_1 = 'data_path1'
    data_path_1 = os.path.join(root_directory_1, 'miss_pixels2013.pkl')
    if os.path.isfile(data_path_1):
        x = pickle.load(open( data_path_1, "rb" ))
    else:
        raise ValueError('Check file {} exist'.format(data_path1))
    bags = [bag[:,1:] for bag in x[:]]
    indexes = [bag[:,0] for bag in x[:]]
    return bags, indexes

def load_malaria(dataset_name, pre_compute_dist=False):
    root_directory_1 = 'data_path1'
    root_directory_2 = 'data_path2'

    if dataset_name == 'malaria_13':
        x_data_path_1 = os.path.join(root_directory_1, 'pixels2013.pkl')
        y_data_path_1 = os.path.join(root_directory_1, 'bag2013.pkl')
        x_data_path_2 = os.path.join(root_directory_2, 'pixels2013.pkl')
        y_data_path_2 = os.path.join(root_directory_2, 'bag2013.pkl')
    else:
        raise ValueError("Please specify malaria_13")
    if os.path.isfile(x_data_path_1) and os.path.isfile(y_data_path_1):
        x = pickle.load(open( x_data_path_1, "rb" ))
        y = pickle.load(open( y_data_path_1, "rb" ))
    elif os.path.isfile(x_data_path_2) and os.path.isfile(y_data_path_2):
        x = pickle.load(open( x_data_path_2, "rb" ))
        y = pickle.load(open( y_data_path_2, "rb" ))
    else:
        raise ValueError("one of or both {} {} {} {}does not exist".format(x_data_path_1, y_data_path_1,
                                                                           x_data_path_2, y_data_path_2,))
    if pre_compute_dist:
        raise NotImplementedError()
        location_path = os.path.join(root_directory, 'pixel_dist', dataset_name)
        if not os.path.exists(location_path):
            raiseValueError('Check {} exists'.format(location_path))
        for index, bag in enumerate(x[:,:-1]):
            file_name = 'euclid_dist_{}'.format(index)
            save_path = os.path.join(location_path, file_name)
            if not os.path.exists(save_path):
                pairwise_dists = squareform(pdist(bag, 'sqeuclidean'))
                np.save(save_path, pairwise_dists)
            else:
                print('{} exists already'.format(file_name))
        print('pre-computed pixel distances at {}'.format(save_path))
        #cell_id, covariates, x, y , pixel_pop
        #bags (x,y at end), indiv_pop, indexes, labels, bag_pop
    return [bag[:,1:-1] for bag in x[:]], [bag[:,-1] for bag in x[:]], [bag[:,0] for bag in x[:]], y[:,1], y[:,0]

def scaler_transform(scaler, data, train=False, coords=False):
    # Stack them and shift xy coordinates if spatial, or transform, then restack.
    if len(data) == 0:
        return scaler, None
    elif scaler is None:
        return scaler, data
    else:
        size_bags = [ len(i) for i in data]
        index_split = np.cumsum(size_bags)[:-1]
        data_stack = np.vstack(data)
        if coords:
            data_main = data_stack[:,:-2]
            data_coords = data_stack[:,-2:]
            data_coords[:,0] = (data_coords[:,0] - 68.4875)/15.11667
            data_coords[:,1] = (data_coords[:,1] - 11.74583)/15.11667
            if train:
                data_stacked_main = scaler.fit_transform(data_main)
            else:
                data_stacked_main = scaler.fit_transform(data_main)
            data_stacked = np.hstack((data_stacked_main, data_coords))
        else:
            if train:
                data_stacked = scaler.fit_transform(np.vstack(data))
            else:
                data_stacked = scaler.transform(np.vstack(data))
        #print(np.max(data_stacked[:,-1]), np.min(data_stacked[:,-1]))
        #print(np.max(data_stacked[:,-2]), np.min(data_stacked[:,-2]))
        data = np.vsplit(data_stacked, index_split) 
    # Checked
    return scaler, data


@contextmanager
# TF session
def tf_session(n_cpus=1, config_args={}, gpu=False, **kwargs):
    import tensorflow as tf
    if gpu:
        config = tf.ConfigProto(log_device_placement=True, **config_args)
    else:
        config = tf.ConfigProto(intra_op_parallelism_threads=n_cpus,
                            inter_op_parallelism_threads=n_cpus, **config_args)
    with tf.Session(config=config) as sess:
        yield sess
