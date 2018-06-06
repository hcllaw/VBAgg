from __future__ import division, print_function
from functools import partial

import numpy as np
import tensorflow as tf

from .base import Network
from vb_utility import make_stable

def build_net(in_dim, n_hidden, data_type, link='exp',
              reg_indiv=0.0, reg_bag=0.0, reg_out=0.0, total_size=None,
              bw_indiv=1.0, bw_bag=1.0, bw_scale=1.0, bw_bag_scale=1.0,
              approx_kernel='rff', kernel='rbf', n_rff=500, var_init=0.01,
              bag_reg=False, indiv_y_bol=False, bw_indiv_L=None,
              seed=23, dtype=tf.float32, log_y=False, **others):
    
    net = Network(in_dim, data_type, n_hidden=n_hidden, link=link, kernel=kernel,
                  approx_kernel=approx_kernel, n_rff=n_rff, var_init=var_init,
                  indiv_bol=indiv_y_bol, dtype=dtype, 
                  bag_reg=bag_reg, seed=seed, log_y=log_y)

    inputs = net.inputs
    params = net.params
    cst = partial(tf.cast, dtype=dtype)
    # Model parameters
    #initializer = tf.initializers.random_normal(seed=seed, dtype=dtype) # normal initialiser 
    initializer = tf.keras.initializers.he_normal(seed=seed)
    params['out'] = tf.Variable(cst(initializer([n_hidden, 1])), name= 'out', dtype=dtype)
    #print('bw_indiv', bw_indiv)
    if kernel in ['ard', 'additive', 'rbf']:
        params['log_bw_sq'] = tf.Variable(2.0*tf.log(tf.constant(bw_indiv, dtype=dtype)), name = 'log_bw_sq')
        #params['log_bw'] = tf.Variable(tf.log(tf.constant(bw_indiv, dtype=dtype)), name = 'log_bw_sq')
        #params['log_scale'] = tf.Variable(tf.constant([0], dtype=dtype), name='log_scale')

    n_bags = cst(tf.shape(inputs['sizes'])[0])
    n_indiv = cst(tf.shape(inputs['X'])[0])
    #params['log_bw'] = tf.Print(params['log_bw'], [params['log_bw']])
    scale = tf.exp(params['log_scale'])

    stddev = tf.sqrt(tf.exp(params['log_bw_sq']))
    #stddev = tf.Print(stddev, [stddev])
    if kernel in ['ard', 'rbf']:
        k_mm = net.kernel(inputs['landmarks'], inputs['landmarks'], stddev=stddev, scale=scale)
        #k_mm = tf.Print(k_mm, [inputs['X']], message='X',summarize=10000)
        k_nm = net.kernel(inputs['X'], inputs['landmarks'], stddev=stddev, scale=scale)
    elif kernel == 'additive':
        scale_mat = tf.exp(params['log_scale_m'])
        k_mm = net.kernel(inputs['landmarks'], inputs['landmarks'], stddev_ard=stddev[:-2], scale_ard=scale, 
                                                                    stddev_mat=stddev[-2:], scale_mat=scale_mat)
        k_nm = net.kernel(inputs['X'], inputs['landmarks'], stddev_ard=stddev[:-2], scale_ard=scale, 
                                                            stddev_mat=stddev[-2:], scale_mat=scale_mat)

    #k_mm = tf.Print(k_mm, [k_mm], message='k_mm', summarize=4)
    k_mm_sqrt_inv = make_stable(k_mm, sqrt=True, inverse=True)
    #k_nm = tf.Print(k_nm, [k_nm], message='k_nm', summarize=5)
    phi = tf.matmul(k_nm, k_mm_sqrt_inv)
    #phi = tf.Print(phi, [phi, params['out']],summarize=50)
    #mean, var = tf.nn.moments(phi,[0])
    #phi = (phi - mean) / tf.sqrt(var)
    #phi = tf.Print(phi, [tf.shape(phi), tf.shape(params['out'])])
    output = tf.squeeze(tf.matmul(phi , params['out']))
    #output = tf.Print(output, [output], 'output')

    net.indiv = indiv = net.linkage(output)
    #indiv = tf.Print(indiv, [indiv])
    net.indiv_nll = net.nll_term(inputs['indiv_y'], indiv)
    net.indiv_se = net.square_err(inputs['indiv_true_y'], indiv)
    #indiv = tf.Print(indiv, [indiv], 'indiv')
    net.indiv_y = indiv_y_pop = tf.multiply(indiv, inputs['indiv_pop']) # [pts]

    indiv = tf.expand_dims(indiv, 1)
    indiv_y_pop = tf.expand_dims(indiv_y_pop, 1)
    net.bag_y = bag_y = tf.squeeze(net.bag_pool(indiv_y_pop))
    #bag_y = tf.Print(bag_y, [bag_y])
    net.bag_nll = l1_term = net.nll_term(inputs['y'], bag_y, bags=True)
    net.bag_se = net.square_err(inputs['y'], bag_y, bags=True)

    l2_term = net.indiv_l2(inputs['X'], output, bw=bw_indiv_L, bw_scale=bw_scale)
    l3_term = net.bag_l3(inputs['X_bag'], output, bw=bw_bag, bw_scale=bw_bag_scale)

    variables = tf.trainable_variables()
    loss_reg = tf.add_n([ tf.nn.l2_loss(v) for v in variables
                          if v.name not in ['log_scale:0','log_bw_sq:0', 'log_sig_sq:0'] ])
    #l2_term = tf.Print(l2_term, [l1_term / n_bags, reg_indiv * l2_term/ (n_indiv**2), 
    #                             reg_bag * l3_term / (n_bags**2), loss_reg * reg_out / n_bags]) # n_bags here should be total_size.... 
    net.loss  = l1_term / n_bags + reg_indiv * l2_term / (n_indiv**2) + reg_bag * l3_term / (n_bags**2) + loss_reg * reg_out/n_bags
    net.check = tf.add_check_numerics_ops()
    return net
