from __future__ import division, print_function
from functools import partial

import numpy as np
import tensorflow as tf

from .base import Network
from .kernel import ard_kernel

def build_net(in_dim, n_hidden, data_type, link='exp',
              reg_indiv=0.0, reg_bag=0.0, reg_out=0.0, var_init=0.01,
              bw_indiv=1.0, bw_bag=1.0, bw_scale=1.0, bw_bag_scale=1.0,
              approx_kernel='rff', n_rff=500, kernel='rbf',
              bag_reg=False, indiv_y_bol=False, landmarks_size=30,
              seed=23, dtype=tf.float32, log_y=False, **others):
    
    net = Network(in_dim, data_type, n_hidden=landmarks_size, link=link, approx_kernel=approx_kernel, kernel=kernel,
                  n_rff=n_rff, indiv_bol=indiv_y_bol, dtype=dtype, var_init=var_init,
                  bag_reg=bag_reg, seed=seed, log_y=log_y, net='nn')
    inputs = net.inputs
    params = net.params

    cst = partial(tf.cast, dtype=dtype)
    # Model parameters
    #initializer = tf.initializers.random_normal(seed=seed, dtype=dtype)
    initializer = tf.keras.initializers.he_normal(seed=seed) # Elevators start using HE, before use Xavier
    #initializer_output = tf.contrib.layers.xavier_initializer(seed=seed)
    z_initializer = tf.zeros_initializer(dtype=dtype)
    params['weights'] = tf.Variable(cst(initializer([in_dim, n_hidden])), name= 'weights', dtype=dtype)
    params['bias'] = tf.Variable(z_initializer([n_hidden]), name = 'bias', dtype=dtype)
    params['out'] = tf.Variable(cst(initializer([n_hidden, 1])), name = 'out', dtype=dtype)
    #Indiviual Model
    n_bags = cst(tf.shape(inputs['sizes'])[0])
    n_indiv = cst(tf.shape(inputs['X'])[0])
    #params['weights'] = tf.Print(params['weights'], [params['weights']], 'weights')
    hidden = tf.nn.relu(tf.matmul(inputs['X'], params['weights']) + params['bias'])
    #params['out'] = tf.Print(params['out'], [params['out']], 'out')
    #hidden = tf.Print(hidden, [hidden], 'hidden')
    output = tf.squeeze(tf.matmul(hidden , params['out'])) # [pts]
    #output = tf.Print(output, [output], 'output')
    net.indiv = indiv = net.linkage(output)
    #indiv = tf.Print(indiv, [indiv, inputs['indiv_y']], 'indiv')
    net.indiv_nll = net.nll_term(inputs['indiv_y'], indiv)
    net.indiv_se = net.square_err(inputs['indiv_true_y'], indiv)

    net.indiv_y = indiv_y_pop = tf.multiply(indiv, inputs['indiv_pop']) # [pts]

    indiv = tf.expand_dims(indiv, 1)
    indiv_y_pop = tf.expand_dims(indiv_y_pop, 1)
    net.bag_y = bag_y = tf.squeeze(net.bag_pool(indiv_y_pop))
    net.bag_nll = l1_term = net.nll_term(inputs['y'], bag_y, bags=True)
    net.bag_se = net.square_err(inputs['y'], bag_y, bags=True)

    l2_term = net.indiv_l2(inputs['X'], output, bw=bw_indiv, bw_scale=bw_scale)
    l3_term = net.bag_l3(inputs['X_bag'], output, bw=bw_bag, bw_scale=bw_bag_scale)
    #Loss
    variables = tf.trainable_variables()
    loss_reg = tf.add_n([ tf.nn.l2_loss(v) for v in variables
                          if v.name not in ['bias:0', 'log_sig_sq:0'] ])
    #l2_term = tf.Print(l2_term, [l1_term / n_bags, reg_indiv * l2_term/ (n_indiv**2), 
    #                             reg_bag * l3_term / (n_bags**2),reg_out *loss_reg])
    net.loss  = l1_term / n_bags + reg_indiv * l2_term / (n_indiv**2) + reg_bag * l3_term / (n_bags**2) #+ loss_reg * reg_out
    net.check = tf.add_check_numerics_ops()
    return net
