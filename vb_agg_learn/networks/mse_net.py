from __future__ import division, print_function

import numpy as np
import tensorflow as tf

from .base import Network

def build_net(in_dim, n_hidden,
              reg_indiv=0.0, reg_bag=0.0, reg_out=0.0, 
              bw_indiv=None, bw_bag=None, bw_scale=1.0, bw_bag_scale=1.0,
              approx_kernel='landmarks', n_rff=200, landmarks=None,
              bag=False, indiv_y_bol=False, bias_out=False, 
              seed=23, dtype=tf.float32):
    net = Network(in_dim, n_hidden, dtype=dtype)
    inputs = net.inputs
    params = net.params

    # Model parameters
    initializer = tf.keras.initializers.he_normal(seed=seed)
    z_initializer = tf.zeros_initializer()
    params['weights'] = tf.Variable(initializer([in_dim, n_hidden]), name= 'weights', dtype=dtype)
    params['bias'] = tf.Variable(z_initializer([n_hidden]), name = 'bias', dtype=dtype)
    params['out'] = tf.Variable(initializer([n_hidden, 1]), name = 'out', dtype=dtype)
    #Indiviual Model
    n_bags = tf.cast(tf.shape(inputs['sizes'])[0], dtype=dtype)
    n_indiv = tf.cast(tf.shape(inputs['X'])[0], dtype=dtype)
    hidden = tf.nn.relu(tf.matmul(inputs['X'], params['weights']) + params['bias'])
    output = tf.squeeze(tf.matmul(hidden , params['out'])) # [pts]
    if bias_out:
        params['bias_out'] = tf.Variable(tf.constant([-8.12], dtype=dtype), name= 'bias_out')
        output = output + params['bias_out'] # [pts]
    rate = tf.exp(output)
    indiv_y_pop = tf.multiply(rate, inputs['indiv_pop']) # [pts]
    net.indiv_y = indiv_y_pop
    # Early stop individual evaluations
    if indiv_y_bol:
        net.indiv_nll = net.bag_nll = tf.constant(0)
        net.indiv_se = tf.reduce_sum(tf.square(indiv_y_pop - inputs['indiv_y']))
    else:
        net.indiv_se = net.indiv_nll = net.bag_nll = tf.constant(0)

    rate = tf.expand_dims(rate, 1)
    indiv_y_pop = tf.expand_dims(indiv_y_pop, 1)
    net.bag_y = bag_y = tf.squeeze(net.bag_pool(indiv_y_pop)) # [bags]
    net.bag_se = l1_term =  tf.reduce_sum(tf.square(bag_y - inputs['y']))
    # True Kernel
    #K_true = rbf_kernel(inputs['X'], inputs['X'], bw_indiv)
    #col_sum_true = tf.diag( tf.reduce_sum(K_true, axis = 1))
    #L_true = col_sum_true - K_true
    #l2_term_true = tf.squeeze(tf.matmul(tf.matmul(tf.transpose(rate), L_true), rate))

    # Fast approximate kernel
    rff_tf = tf.contrib.kernel_methods.RandomFourierFeatureMapper
    rff_mapper = rff_tf(in_dim, n_rff, stddev=bw_indiv*bw_scale, seed=23)
    fourier_layer = rff_mapper.map(inputs['X']) # [pts, n_rff] \Phi
    fourier_y = tf.squeeze(tf.matmul(tf.transpose(fourier_layer), rate)) # \Phi^\top \lambda
    l2_term_0 = tf.reduce_sum(tf.multiply(fourier_y, fourier_y)) # square l2 norm of \Phi^\top \lambda
    rff_rowsum = tf.reduce_sum(fourier_layer, axis=0, keep_dims=True) # \mathcal{1} \Phi, row 
    pre_diag = tf.squeeze(tf.matmul(fourier_layer, tf.transpose(rff_rowsum))) #\Phi \Phi^\top \mathcal{1}^\top
    diag_colsum = tf.diag(pre_diag) # construct diag 
    #\lambda^\top diag( \Phi \Phi^\top \mathcal{1}^\top) \lambda
    l2_term_1 = tf.squeeze(tf.matmul(tf.matmul(tf.transpose(rate), diag_colsum), rate))
    l2_term = l2_term_1 - l2_term_0

    #bag_loss = np.square(bag_rate - tf.tranpose(bag_rate)) # form bag loss matrix
    
    #l3_term = tf.reduce_sum(tf.multiply(inputs['bag_kernel'], bag_loss)) # multiply and sum Laplacian Regulariser-Bag
    #Loss
    variables = tf.trainable_variables()

    loss_reg = tf.add_n([ tf.nn.l2_loss(v) for v in variables
                    if 'bias' not in v.name ]) * reg_out

    net.loss  = l1_term / n_bags #+ reg_indiv * l2_term / (n_indiv**2) + loss_reg #+ reg_bag * l3_term
    # Can add net.print_out
    return net
