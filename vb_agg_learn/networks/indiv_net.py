from __future__ import division, print_function
from math import pi

import numpy as np
import tensorflow as tf

from .base import Network

def normal_likelihood(data_y, mean_y, log_bw_sq, bag_size=None, dtype=tf.float32):
    if bag_size is None:
        size = tf.shape(data_y)
        bag_size = tf.ones(size, dtype=dtype)
    constant = 0.5 * tf.log(2.0 * pi * tf.exp(log_bw_sq) * bag_size)
    main = tf.div(tf.square(data_y - mean_y), 2.0 * bag_size * tf.exp(log_bw_sq))
    nll = tf.reduce_sum(constant + main)
    return nll 

def build_net(in_dim, n_hidden, reg_out=0.0, seed=23, dtype=tf.float32, **others):
    net = Network(in_dim, n_hidden, dtype=dtype)
    inputs = net.inputs
    params = net.params
    # Model parameters
    initializer = tf.keras.initializers.he_normal(seed=seed)
    z_initializer = tf.zeros_initializer()
    params['weights'] = tf.Variable(initializer([in_dim, n_hidden]), name= 'weights', dtype=dtype)
    params['bias'] = tf.Variable(z_initializer([n_hidden]), name = 'bias', dtype=dtype)
    params['out'] = tf.Variable(initializer([n_hidden, 1]), name = 'out', dtype=dtype)
    params['log_bw_sq'] = tf.Variable(tf.random_normal([1], seed=seed, dtype=dtype), name = 'log_bw_sq')

    #Indiviual Model
    hidden = tf.nn.relu(tf.matmul(inputs['X'], params['weights']) + params['bias'])
    indiv_y = tf.squeeze(tf.matmul(hidden, params['out'])) # [pts]
    n_indiv = tf.cast(tf.shape(inputs['X'])[0], dtype=dtype)
    #Early Stop individual evaluations 
    net.indiv_nll = indiv_nll = normal_likelihood(inputs['indiv_y'], indiv_y, 
                                        params['log_bw_sq'], dtype=dtype)
    
    net.indiv_se = tf.reduce_sum(tf.square(indiv_y - inputs['indiv_y']))
    net.indiv_y = indiv_y

    indiv_y = tf.expand_dims(indiv_y, 1)
    net.bag_y = bag_y = tf.squeeze(net.bag_pool(indiv_y)) 

    net.bag_nll = l1_term = normal_likelihood(inputs['y'], bag_y, params['log_bw_sq'], 
                                              bag_size=inputs['sizes'], dtype=dtype)
    net.bag_se = tf.reduce_sum(tf.square(bag_y - inputs['y']))

    variables = tf.trainable_variables()
    loss_reg = tf.add_n([ tf.nn.l2_loss(v) for v in variables
                    if 'bias' not in v.name ]) * reg_out
    
    net.loss  = indiv_nll / n_indiv + loss_reg #+ reg_bag * l3_term
    return net
