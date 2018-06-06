from __future__ import division, print_function
from math import pi
from functools import partial

import numpy as np
import tensorflow as tf

from .base import Network

def build_net(in_dim, n_hidden, data_type, link='exp',
              reg_out=0.0, seed=23, dtype=tf.float32, var_init=0.01,
              indiv_y_bol=False, **others):
    
    net = Network(in_dim, data_type, n_hidden=n_hidden, link=link, var_init=var_init,
                  dtype=dtype, indiv_bol=indiv_y_bol, seed=seed)
    inputs = net.inputs
    params = net.params
    # Model parameters

    cst = partial(tf.cast, dtype=dtype)
    n_bags = cst(tf.shape(inputs['sizes'])[0])
    initializer = tf.keras.initializers.he_normal(seed=seed)
    z_initializer = tf.zeros_initializer()
    params['weights'] = tf.Variable(cst(initializer([in_dim, n_hidden])), name= 'weights', dtype=dtype)
    params['bias'] = tf.Variable(z_initializer([n_hidden]), name = 'bias', dtype=dtype)
    params['out'] = tf.Variable(cst(initializer([n_hidden, 1])), name = 'out', dtype=dtype)
    
    #Indiviual Model (learnt from bag model)
    bag_x = net.bag_pool(tf.multiply(inputs['X'], tf.expand_dims(inputs['weights'], 1)))  # check the broadcasting here!
    #bag_x = tf.Print(bag_x, [tf.expand_dims(inputs['weights'], 1)])
    hidden = tf.nn.relu(tf.matmul(bag_x, params['weights']) + params['bias'])
    output = tf.squeeze(tf.matmul(hidden, params['out'])) # [pts]
    bag_fake_y = net.linkage(output)

    if indiv_y_bol:
        hidden_indiv = tf.nn.relu(tf.matmul(inputs['X'], params['weights']) + params['bias'])
        indiv_output = tf.squeeze(tf.matmul(hidden_indiv, params['out'])) # [pts]
        net.indiv = indiv = net.linkage(indiv_output)
        net.indiv_y = indiv_y = tf.multiply(indiv, inputs['indiv_pop']) # [pts]
        net.indiv_nll = net.nll_term(inputs['indiv_y'], indiv)
        net.indiv_se = net.square_err(inputs['indiv_true_y'], indiv)

    net.bag_y = bag_y = tf.multiply(bag_fake_y, inputs['bag_pop'])
    #net.indiv_nll = net.nll_term(inputs['indiv_y'], indiv)
    #net.indiv_se = net.square_err(inputs['indiv_y'], indiv)

    net.bag_nll = l1_term = net.nll_term(inputs['y'], bag_y, bags=True, baseline=True)
    net.bag_se = net.square_err(inputs['y'], bag_y, bags=True)

    variables = tf.trainable_variables()
    loss_reg = tf.add_n([ tf.nn.l2_loss(v) for v in variables
                    if v.name not in ['bias:0', 'log_sig_sq:0']]) 
    net.loss  = l1_term / n_bags + loss_reg * reg_out
    return net
