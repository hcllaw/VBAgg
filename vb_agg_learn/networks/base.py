from __future__ import division
from collections import namedtuple
from functools import partial
import abc

import numpy as np
import tensorflow as tf
from math import log, pi

from .kernel import ard_kernel, ard_matern_kernel

SparseInfo = namedtuple('SparseInfo', ['indices', 'values', 'dense_shape'])

# Tensorflow normal likelihoof
def normal_likelihood(data_y, mean_y, log_bw_sq, bag_size=None, dtype=tf.float32):
    if bag_size is None:
        size = tf.shape(data_y)
        bag_size = tf.ones(size, dtype=dtype)
    constant = 0.5 * tf.log(2.0 * pi * tf.exp(log_bw_sq) * bag_size)
    main = tf.div(tf.square(data_y - mean_y), 2.0 * bag_size * tf.exp(log_bw_sq))
    nll = tf.reduce_sum(constant + main)
    return nll 
# For sparse matrix 
def sparse_matrix_placeholders(dtype=np.float32):
    '''
    Placeholders for a tf.SparseMatrix; use tf.SparseMatrix(*placeholders)
    to make the actual object.
    '''
    return SparseInfo(
        indices=tf.placeholder(tf.int64, [None, 2]),
        values=tf.placeholder(dtype, [None]),
        dense_shape=tf.placeholder(tf.int64, [2]),
    )

# indicies are a list of [[1,3], [6,7]] where means place elements in (1,3) and (6,7) entry in 2D matrix
# values are a list of dtype placing at those entries above [ 3, 4] place at (1,3) and (6,7)
# Specify dimension of tensor, here might be [7, 7] for size 7 by 7
# For sum pooling
def mean_matrix(feats, sparse=False, dtype=np.float32):
    '''
    Returns a len(feats) x feats.total_pts matrix to do sum pooling by bag.
    '''
    if sparse:
        bounds = np.r_[0, np.cumsum(feats.n_pts)]
        return SparseInfo(
            indices=np.vstack([
                [i, j]
                for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:]))
                for j in range(start, end)
            ]),
            values=[1 for bag in feats for _ in range(len(bag))],
            dense_shape=[len(feats), feats.total_points],
        )
    else:
        mean_mat = np.zeros((len(feats), feats.total_points), dtype=dtype)
        index = 0
        for j in range(len(feats)):
            index_up = index
            index_down = index + feats.n_pts[j]
            mean_mat[j, index_up:index_down] = 1
            index = index_down
        return mean_mat

# Network class
class Network(object):
    def __init__(self, in_dim, data_type, n_hidden=50, link='exp', approx_kernel='rff', var_init=None,
                 kernel='rbf', n_rff=100, indiv_bol=False, bag_reg=False, dtype=tf.float32, seed=23, 
                 log_y=False, ard_mat_init_scale=0.5, net=None):
        self.in_dim = in_dim
        self.data_type = data_type
        self.link = link
        self.log_y = log_y
        self.inputs = {
            'X': tf.placeholder(dtype, [None, in_dim]),  # all bags stacked up
            'sizes': tf.placeholder(dtype, [None]),  # one per bag
            'mean_matrix': sparse_matrix_placeholders(dtype),  # n_bags, n_pts_batch
            'indiv_true_y': tf.placeholder(dtype, [None]), # one per individual
            'indiv_y': tf.placeholder(dtype, [None]), # one per individual
            'X_bag' : tf.placeholder(dtype, [None, None]), # one per bag
            'weights': tf.placeholder(dtype, [None]),
            'landmarks': tf.placeholder(dtype, [n_hidden, in_dim]),
            'y': tf.placeholder(dtype, [None]),  # one per bag # Observed y
            'indiv_pop': tf.placeholder(dtype, [None]),  # one per individual
            'bag_pop': tf.placeholder(dtype, [None]), # one per bag
            'bag_true_y': tf.placeholder(dtype, [None]), # one per bag
        }
        self.params = {}
        if self.data_type == 'normal':
            # Change accordingly...
            self.params['log_sig_sq'] = tf.Variable(tf.random_normal([1], mean=log(var_init), stddev=0.5,
                                                    seed=seed, dtype=dtype), name = 'log_sig_sq')
        self.net = net
        self.dtype = dtype
        self.approx_kernel = approx_kernel
        self.n_rff = n_rff
        self.indiv_bol = indiv_bol
        self.bag_reg = bag_reg
        self.kernel_name = kernel
        # Scaling for kernel
        if kernel in ['rbf', 'ard']:
            self.params['log_scale'] = tf.Variable(tf.constant([0], dtype=dtype), name='log_scale')
            self.kernel = ard_kernel
        elif kernel == 'additive':
            self.params['log_scale'] = tf.Variable(tf.constant([log(ard_mat_init_scale)],dtype=dtype), name='log_scale')
            self.params['log_scale_m'] = tf.Variable(tf.constant([log(ard_mat_init_scale)],dtype=dtype), name='log_scale_m')
            self.kernel = ard_matern_kernel
        else:
            raise ValueError('{} not implemented'.format(kernel))
    
    def square_err(self, true_y, pred_y, bags=False):
        if not bags and not self.indiv_bol:
            se = tf.cast(tf.constant(0), self.dtype)
        else:
            true_y = tf.squeeze(true_y)
            pred_y = tf.squeeze(pred_y)
            if self.log_y: # zero log issue
                zeros = tf.zeros_like(true_y, dtype=self.dtype)
                condition_tr = tf.greater(true_y, zeros)
                condition_pr = tf.greater(pred_y, zeros)
                case_false = tf.ones_like(true_y, dtype=self.dtype)
                true_y = tf.where(condition_tr, true_y, case_false)
                pred_y = tf.where(condition_pr, pred_y, case_false)
                true_y = tf.log(true_y)
                pred_y = tf.log(pred_y)
            se = tf.reduce_sum(tf.square(true_y - pred_y))
        return se

    def nll_term(self, true_y, pred_y, bags=False, baseline=False):
        if not bags and not self.indiv_bol:
            nll = tf.cast(tf.constant(0), self.dtype)
        else:
            true_y = tf.squeeze(true_y)
            pred_y = tf.squeeze(pred_y)
            if self.data_type == 'poisson':
                pois_loss = tf.nn.log_poisson_loss
                nll = tf.reduce_sum(pois_loss(true_y, tf.log(pred_y), 
                                              compute_full_loss=True))
            elif self.data_type == 'normal':
                if baseline:
                    bag_size = tf.square(self.inputs['sizes'])
                else:
                    bag_size = self.inputs['sizes']
                if bags:
                    nll = normal_likelihood(true_y, pred_y, self.params['log_sig_sq'], 
                                            bag_size=bag_size, dtype=self.dtype)
                else:
                    nll = normal_likelihood(true_y, pred_y, self.params['log_sig_sq'], 
                                            dtype=self.dtype)
        nll = tf.reduce_sum(nll)
        return nll

    def linkage(self, output):
        if self.data_type == 'poisson':
            if self.link == 'exp':
                return tf.exp(output)
            elif self.link == 'square':
                return tf.square(output)
        elif self.data_type == 'normal':
            return output
        else:
            raise ValueError('{} not available data type'.format(self.data_type))
    # Bag Laplacian Kernel
    def bag_l3(self, inputs_X, output, bw=1.0, bw_scale=1.0):
        if self.link == 'square': # As we want it to be smooth correctly vs the Laplacian for square
            output = tf.abs(output) 
        if self.bag_reg: 
            stddev = bw * bw_scale
            output = tf.expand_dims(output, 1)
            bag_output = tf.squeeze(self.bag_pool(output))
            stand_y = tf.divide(bag_output, self.inputs['sizes'])
            stand_y = tf.expand_dims(stand_y, 1) 
            K_bag_true = ard_kernel(inputs_X, inputs_X, stddev=stddev)
            bag_col_sum = tf.diag( tf.reduce_sum(K_bag_true, axis = 1))
            L_bag = bag_col_sum - K_bag_true
            l3_term = tf.squeeze(tf.matmul(tf.matmul(tf.transpose(stand_y), L_bag), stand_y))
        else:
            l3_term = tf.cast(tf.constant(0), self.dtype)
        return l3_term
    # Individual Laplacian Kernel 
    def indiv_l2(self, inputs_X, indiv_y, bw=1.0, bw_scale=1.0):
        stddev = bw_scale * bw
        indiv_y = tf.expand_dims(indiv_y, 1)
        landmarks = self.inputs['landmarks']
        if self.link == 'square': # As we want it to be smooth correctly vs the Laplacian for square
            indiv_y = tf.abs(indiv_y) 
            #print('square link')
        # Landmark Laplacian Kernel for NN additive 
        if self.kernel_name == 'additive' and self.net == 'nn':
            #print('Using additive Kernel')
            K_mn = self.kernel(landmarks, inputs_X, stddev_ard=stddev[:-2], scale_ard=0.5, 
                                                   stddev_mat=stddev[-2:], scale_mat=0.5)
            K_mm = self.kernel(landmarks, landmarks, stddev_ard=stddev[:-2], scale_ard=0.5, 
                                                    stddev_mat=stddev[-2:], scale_mat=0.5)
            K_mm_inv = tf.matrix_inverse(K_mm)
            feat_indiv_y = tf.matmul(K_mn, indiv_y)
            l2_term_0 = tf.squeeze(tf.matmul(tf.matmul(tf.transpose(feat_indiv_y), K_mm_inv), feat_indiv_y))
            colsum = tf.reduce_sum(K_mn, axis=1, keep_dims=True)
            diag_colsum = tf.squeeze(tf.matmul(tf.matmul(tf.transpose(K_mn), K_mm_inv), colsum))
            l2_term_1 = tf.reduce_sum(tf.multiply(tf.multiply(diag_colsum, tf.squeeze(indiv_y)), tf.squeeze(indiv_y)))

            #l2_term_1 = tf.squeeze(tf.matmul(tf.matmul(tf.transpose(indiv_y), diag_colsum), indiv_y))
            l2_term = l2_term_1 - l2_term_0
        else:
            # Use RFF Laplacian Kernel for others
            stddev = bw * bw_scale
            rff_tf = tf.contrib.kernel_methods.RandomFourierFeatureMapper
            rff_mapper = rff_tf(self.in_dim, self.n_rff, stddev=stddev, seed=23)
            # RFF does not support float64
            features = rff_mapper.map(tf.cast(inputs_X, tf.float32))
            features = tf.cast(features, self.dtype)
            feat_indiv_y = tf.squeeze(tf.matmul(tf.transpose(features), indiv_y)) # \Phi^\top \lambda
            l2_term_0 = tf.reduce_sum(tf.multiply(feat_indiv_y, feat_indiv_y)) # square l2 norm of \Phi^\top \lambda
            rowsum = tf.reduce_sum(features, axis=0, keep_dims=True) # \mathcal{1} \Phi, row 
            pre_diag = tf.squeeze(tf.matmul(features, tf.transpose(rowsum))) #\Phi \Phi^\top \mathcal{1}^\top
            #diag_colsum = tf.diag(pre_diag) # construct diag
            l2_term_1 = tf.reduce_sum(tf.multiply(tf.multiply(pre_diag, tf.squeeze(indiv_y)), tf.squeeze(indiv_y)))
            #\lambda^\top diag( \Phi \Phi^\top \mathcal{1}^\top) \lambda
            #l2_term_1 = tf.squeeze(tf.matmul(tf.matmul(tf.transpose(indiv_y), diag_colsum), indiv_y))
            #l2_term_1 = tf.Print(l2_term_1, [l2_term_1, l2_term_1_n])
            l2_term = l2_term_1 - l2_term_0
        return l2_term
    # Feeding into network
    def feed_dict(self, batch, labels=None, landmarks=None):
        batch.make_stacked()
        i = self.inputs
        stack_batch = batch.stacked_features
        #print(batch.total_points)
        if batch.pop and batch.indiv and batch.true_indiv:
            data = stack_batch[:,:-3]
            pop = stack_batch[:,-3]
            true_indiv = stack_batch[:,-2]
            indiv = stack_batch[:,-1]
        elif batch.pop and batch.indiv:
            data = stack_batch[:,:-2]
            pop = stack_batch[:,-2]
            indiv = stack_batch[:,-1]
        elif batch.pop:
            data = stack_batch[:,:-1]
            pop = stack_batch[:,-1]
        elif batch.indiv:
            raise NotImplementedError()
            data = stack_batch[:,:-2]
            true_indiv = stack_batch[:,-2]
            indiv = stack_batch[:,-1]
            pop = np.ones(batch.total_points)
        else:
            data = stack_batch
            pop = np.ones(batch.total_points)
        sizes = batch.n if hasattr(batch, 'n') else batch.n_pts
        index = [0] + np.cumsum(sizes).tolist()
        bag_pop = np.array([ np.sum(pop[ index[l]:index[l+1] ]) for l in range(len(sizes))])
        weights = np.divide(pop, np.repeat(bag_pop, sizes))
        d = { i['X']: data,
            i['sizes']: sizes,
            i['indiv_pop']: pop,
            i['weights']: weights,
            i['bag_pop']: bag_pop,
        }
        for p, v in zip(i['mean_matrix'], mean_matrix(batch, sparse=True)):
            d[p] = v
        if hasattr(batch, 'bag_var'):
            d[i['X_bag']] = batch.bag_var
        if hasattr(batch, 'true_y'):
            d[i['bag_true_y']] = batch.true_y
        if batch.true_indiv and batch.indiv:
            d[i['indiv_true_y']] = true_indiv
            d[i['indiv_y']] = indiv
        elif batch.indiv:
            d[i['indiv_true_y']] = indiv
            d[i['indiv_y']] = indiv
        if labels is not None:
            d[i['y']] = labels
        if landmarks is not None:
            d[i['landmarks']] = landmarks
        return d

    def bag_pool(self, layer):
        mean_matrix = tf.SparseTensor(*self.inputs['mean_matrix'])
        return tf.sparse_tensor_dense_matmul(mean_matrix, layer)

'''
if self.approx_kernel == 'landmarks':
    landmarks = tf.convert_to_tensor(self.landmarks, dtype=self.dtype)
    K_mn = tf.transpose(ard_kernel(inputs_X, landmarks, stddev=stddev))
    K_mm = ard_kernel(landmarks, landmarks, stddev=stddev)
    s, u, v = tf.svd(K_mm)
    #s = tf.Print(s, [s], summarize=1000)
    si = tf.where(tf.less_equal(s, 0), tf.tile(tf.constant([1e-8], dtype=self.dtype), [tf.shape(s)[0]]), s)
    #si = tf.Print(si, [si], summarize=1000)
    K_mm_stable = tf.matmul(tf.matmul(u, tf.diag(si)), tf.transpose(v))
    #K_mm_stable = tf.Print(K_mm_stable, [tf.shape(K_mm_stable)])
    K_mm_inv = tf.matrix_inverse(K_mm_stable)
    feat_indiv_y = tf.matmul(K_mn, indiv_y)
    l2_term_0 = tf.squeeze(tf.matmul(tf.matmul(tf.transpose(feat_indiv_y), K_mm_inv), feat_indiv_y))
    colsum = tf.reduce_sum(K_mn, axis=1, keep_dims=True)
    diag_colsum = tf.diag(tf.squeeze(tf.matmul(tf.matmul(tf.transpose(K_mn), K_mm_inv), colsum)))
    l2_term_1 = tf.squeeze(tf.matmul(tf.matmul(tf.transpose(indiv_y), diag_colsum), indiv_y))
    l2_term = l2_term_1 - l2_term_0
elif self.approx_kernel == 'rff':

    # Others
# True Kernel
#K_true = rbf_kernel(inputs['X'], inputs['X'], bw_indiv)
#col_sum_true = tf.diag( tf.reduce_sum(K_true, axis = 1))
#L_true = col_sum_true - K_true
#l2_term_true = tf.squeeze(tf.matmul(tf.matmul(tf.transpose(indiv_y), L_true), indiv_y))

# Slow Approximate Kernel
#K = tf.matmul(fourier_layer, tf.transpose(fourier_layer))
#col_sum = tf.diag( tf.reduce_sum(K, axis = 1)) # sum across rows
#L = col_sum - K
#l2_term_slow = tf.squeeze(tf.matmul(tf.matmul(tf.transpose(indiv_y), L), indiv_y))


def indiv_l2_check(self, inputs_X, indiv_y, stddev=1.0):
    K_true = ard_kernel(inputs_X, inputs_X, stddev=stddev)
    col_sum_true = tf.diag( tf.reduce_sum(K_true, axis = 1))
    L_true = col_sum_true - K_true
    l2_term_true = tf.squeeze(tf.matmul(tf.matmul(tf.transpose(indiv_y), L_true), indiv_y))
    return l2_term_true
'''
