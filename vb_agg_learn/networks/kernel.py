# kernels
from __future__ import division
from math import sqrt

import tensorflow as tf
import numpy as np
import sklearn.metrics.pairwise as sk_pair
#from agg_learn.utils import symmetric_matrix_square_root
'''
def rbf_kernel(X, Y, stddev=1.0):
    stddev = tf.convert_to_tensor(stddev, X.dtype)
    gamma = 1.0 / (2.0 * tf.square(stddev))
    X_sqnorms_row = tf.expand_dims(tf.reduce_sum(tf.square(X), 1), 1)
    Y_sqnorms = tf.reduce_sum(tf.square(Y), 1)
    Y_sqnorms_col = tf.expand_dims(Y_sqnorms, 0)
    #if return_features:
    #    Y_sqnorms_row = tf.expand_dims(Y_sqnorms, 1)
    #    YY = tf.matmul(Y, Y, transpose_b=True)
    #    XY = tf.matmul(X, Y, transpose_b=True)
    #    K_yy = tf.exp(-gamma * (-2 * YY + Y_sqnorms_row + Y_sqnorms_col))
    #    K_xy = tf.exp(-gamma * (-2 * XY + X_sqnorms_row + Y_sqnorms_col))
    #    K_yy_inv = tf.matrix_inverse(K_yy)
    #    K_yy_inv = tf.Print(K_yy_inv, [K_yy_inv],  summarize=50, message='inv')
    #    K_yy_inv_sqrt = symmetric_matrix_square_root(K_yy_inv)
    #    K_yy_inv_check = tf.matmul(K_yy_inv_sqrt, K_yy_inv_sqrt)
    #    K_yy_inv_sqrt = tf.Print(K_yy_inv_sqrt, [K_yy_inv_sqrt], summarize=50, message='inv_check')
    #    return tf.matmul(K_xy, K_yy_inv_sqrt)
    #    else:
    XY = tf.matmul(X, Y, transpose_b=True)
    return tf.exp(-gamma * (-2 * XY + X_sqnorms_row + Y_sqnorms_col))
'''
def matern32_kernel(X, Y, stddev=1.0, expand=False, tensorf=True):
    if tensorf:
        if expand:
            X = tf.expand_dims(X, 1)
            Y = tf.expand_dims(Y, 1)
        X = tf.divide(X, stddev)
        Y = tf.divide(Y, stddev)
        #X = tf.Print(X, [tf.shape(X), tf.shape(Y)])
        X_sqnorms_row = tf.expand_dims(tf.reduce_sum(tf.square(X), 1), 1)
        Y_sqnorms_col = tf.expand_dims(tf.reduce_sum(tf.square(Y), 1), 0)
        XY = tf.matmul(X, Y, transpose_b=True)
        dist = tf.sqrt(tf.maximum(-2 * XY + X_sqnorms_row + Y_sqnorms_col,1e-32))
        dist = tf.check_numerics(dist, 'dist')
        value = sqrt(3.0)*dist
        return tf.multiply((1.0 + value), tf.exp(-value))
    else:
        if expand:
            X = np.expand_dims(X, 1)
            Y = np.expand_dims(Y, 1)
        X = np.divide(X, stddev)
        Y = np.divide(Y, stddev)
        #X = tf.Print(X, [tf.shape(X), tf.shape(Y)])
        X_sqnorms_row = np.expand_dims(np.sum(np.square(X), axis=1), axis=1)
        Y_sqnorms_col = np.expand_dims(np.sum(np.square(Y), axis=1), axis=0)
        XY = np.matmul(X, Y.T)
        dist = np.sqrt(np.maximum(-2 * XY + X_sqnorms_row + Y_sqnorms_col, 1e-32))
        value = sqrt(3.0) * dist
        return np.multiply((1.0 + value), np.exp(-value))

def ard_matern_kernel(X, Y, stddev_ard=1.0, 
                      stddev_mat=1.0, scale_ard=1.0, 
                      scale_mat=1.0, tensorf=True):
    # Check tf slice
    # only works for dimension 2, as currently only support seperable kernel
    mat_dim = 2
    ard_X = X[:, :-mat_dim]
    ard_Y = Y[:, :-mat_dim]
    mat_X = X[:, -mat_dim:]
    mat_Y = Y[:, -mat_dim:]
    #mat_part_1 = matern32_kernel(mat_X[:,0], mat_Y[:,0], stddev=stddev_mat[0], expand=True)
    #mat_part_2 = matern32_kernel(mat_X[:,1], mat_Y[:,1], stddev=stddev_mat[1], expand=True)
    mat_part = scale_mat * matern32_kernel(mat_X, mat_Y, stddev=stddev_mat, tensorf=tensorf)
    #e, V = np.linalg.eig(mat_part)
    #print('e', e)
    #mat_part = tf.Print(mat_part, [mat_part], 'mat_part', summarize=1000)
    #mat_part = tf.check_numerics(mat_part, 'mat_part')
    #mat_part = scale_mat * tf.multiply(mat_part_1, mat_part_2)
    ard_part = scale_ard * ard_kernel(ard_X, ard_Y, stddev=stddev_ard, tensorf=tensorf)
    #ard_part = tf.Print(ard_part, [ard_part, mat_part])
    return ard_part + mat_part

def ard_kernel(X, Y, stddev=1.0, scale=1.0, tensorf=True):
    if tensorf:
        #X = tf.Print(X, [X, stddev], summarize=5, message ='before')
        X = tf.divide(X, stddev)
        Y = tf.divide(Y, stddev)
        #X = tf.Print(X, [X], summarize=5, message ='Xafter')
        #Y = tf.Print(Y, [Y], summarize=30, message ='Yafter')
        X_sqnorms_row = tf.expand_dims(tf.reduce_sum(tf.square(X), 1), 1)
        #X_sqnorms_row = tf.Print(X_sqnorms_row, [X_sqnorms_row], message= 'X_sqnorms_row')
        Y_sqnorms_col = tf.expand_dims(tf.reduce_sum(tf.square(Y), 1), 0)
        XY = tf.matmul(X, Y, transpose_b=True)
        #XY = tf.Print(XY, [XY], message= 'XY')
        #XY = tf.Print(XY, [-0.5 * (-2 * XY + X_sqnorms_row + Y_sqnorms_col)], summarize=30, message='XY')
        return scale * tf.exp(-0.5 * (-2 * XY + X_sqnorms_row + Y_sqnorms_col))
    else:
        X = np.divide(X, stddev)
        Y = np.divide(Y, stddev)
        X_sqnorms_row = np.expand_dims(np.sum(np.square(X), axis=1), axis=1)
        Y_sqnorms_col = np.expand_dims(np.sum(np.square(Y), axis=1), axis=0)
        XY = np.matmul(X, Y.T)
        return scale * np.exp(-0.5 * (-2 * XY + X_sqnorms_row + Y_sqnorms_col))

