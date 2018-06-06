# Variational optimisation
from __future__ import division, print_function
from functools import partial

import numpy as np
import tensorflow as tf
from math import log, pi, sqrt

from .base import Network
from vb_utility import make_stable, triangular_vec, fill_triangular, term_1_func_additive, term_1_func

def build_net(in_dim, n_hidden, data_type, link='square', total_size=None,
              bw_indiv=1.0, indiv_y_bol=False, kernel='ard', initialse='identity',
              seed=23, dtype=tf.float32, landmarks=None, log_y=False,  device_name=None,
              avg_label=1.0, **others):
    with tf.device(device_name): 
        if avg_label - 1.0 < 0: # HACK FIX FOR MORE GENERAL DATA.
            #print('Alternate Intialisation')
            ard_mat_init_scale = 0.15 # For malaria
            mean_scale = sqrt(avg_label - ard_mat_init_scale*2.0)
        else:
            mean_scale = sqrt(avg_label - 1.0) # i.e. predict baseline at start.
            ard_mat_init_scale = 0.5
        net = Network(in_dim, data_type, n_hidden=n_hidden, link=link, kernel=kernel,
                      indiv_bol=indiv_y_bol, dtype=dtype, seed=seed, log_y=log_y, 
                      ard_mat_init_scale=ard_mat_init_scale)
        inputs = net.inputs
        params = net.params
        land_size = n_hidden
        cst = partial(tf.cast, dtype=dtype)
        # Model parameters
        initializer = tf.initializers.random_normal(seed=seed, dtype=dtype) # normal initialiser 
        z_initializer = tf.zeros_initializer(dtype=dtype)
        o_initializer = tf.ones_initializer(dtype=dtype)
        #initializer = tf.keras.initializers.he_normal(seed=seed)
        if initialse == 'identity':
            triangle_vec = tf.constant(triangular_vec(None, n=land_size), dtype=dtype)
        elif initialse == 'kernel':
            if kernel == 'additive':
                init_kernel = net.kernel(landmarks, landmarks, stddev_ard=bw_indiv[:-2], scale_ard=ard_mat_init_scale, 
                                                               stddev_mat=bw_indiv[-2:], scale_mat=ard_mat_init_scale, 
                                                               tensorf=False)
            elif kernel in ['rbf', 'ard']:
                init_kernel = net.kernel(landmarks, landmarks, stddev=bw_indiv, scale=1.0, tensorf=False)
            L = np.linalg.cholesky(init_kernel)
            #print('L', L)
            triangle_vec = tf.constant(triangular_vec(L, n=land_size), dtype=dtype)
        # Intialise with L = I for safe inversion at start.
        #print('bw_indiv', bw_indiv)
        #print('mean_scale', mean_scale)
        params['L'] = tf.Variable(triangle_vec, name= 'L', dtype=dtype)
        params['mean'] = tf.Variable(mean_scale * o_initializer([land_size, 1]), name = 'mean', dtype=dtype)
        params['prior_mean'] = tf.Variable(z_initializer([1]), name = 'prior_mean', dtype=dtype)

        if kernel in ['ard', 'additive']:
            params['log_bw'] = tf.Variable(tf.log(tf.constant(bw_indiv, dtype=dtype)), name = 'log_bw_sq')
        elif kernel == 'rbf':
            #print('Vary Bandwidth RBF')
            params['log_bw'] = tf.Variable(tf.log(tf.constant(bw_indiv, dtype=dtype)), name = 'log_bw_sq')

        n_bags = cst(tf.shape(inputs['sizes'])[0])
        n_indiv = cst(tf.shape(inputs['X'])[0])

        scale = tf.exp(params['log_scale'])
        stddev = tf.exp(params['log_bw'])
        
        landmarks = inputs['landmarks']
        #stddev = tf.Print(stddev, [stddev], message='bw', summarize=100)
        if kernel in ['ard', 'rbf']:
            k_ww = net.kernel(landmarks, landmarks, stddev=stddev, scale=scale)
            k_wz = net.kernel(landmarks, inputs['X'], stddev=stddev, scale=scale) #K_wz
            #k_wz = tf.Print(k_wz, [k_wz])
            term_0_diag = scale * tf.ones([tf.cast(n_indiv, dtype=tf.int32)], dtype=dtype) #k_zz diagonal
        elif kernel == 'additive':
            scale_mat = tf.exp(params['log_scale_m'])
            k_ww = net.kernel(landmarks, landmarks, stddev_ard=stddev[:-2], scale_ard=scale, 
                                                    stddev_mat=stddev[-2:], scale_mat=scale_mat)
            k_wz = net.kernel(landmarks, inputs['X'], stddev_ard=stddev[:-2], scale_ard=scale, 
                                                      stddev_mat=stddev[-2:], scale_mat=scale_mat)
            term_0_diag = (scale + scale_mat) * tf.ones([tf.cast(n_indiv, dtype=tf.int32)], dtype=dtype) 
        
        chol_k = tf.cholesky(k_ww)
        k_ww_inv = tf.matrix_inverse(k_ww) # K_ww^-1
        triangular = fill_triangular(params['L']) #\Sigma_u=LL^T
        Sigma_u = tf.matmul(triangular, tf.transpose(triangular)) # Sigma_u = L L^T

        k_inv_k_wz = tf.matmul(k_ww_inv, k_wz) # K_ww^-1 K_wz
        mean_diff = params['mean'] - params['prior_mean']
        # mu_prior + K_zw K_ww^-1 (mu_u - mu_prior)
        net.mu = mu = params['prior_mean'] + tf.squeeze(tf.matmul(tf.transpose(k_inv_k_wz), mean_diff)) 

        inputs_int = tf.concat([tf.constant([0], tf.int32), tf.cumsum(tf.cast(inputs['sizes'], tf.int32))], 0)
        if kernel in ['ard', 'rbf']:
            term_1_vec = tf.map_fn(fn=lambda k: term_1_func(net, mu, inputs, stddev, scale, k_wz, Sigma_u,
                                                            inputs_int[k], inputs_int[k+1], k_inv_k_wz),
                                       elems=tf.range(tf.cast(n_bags, dtype=tf.int32)),
                                       dtype=dtype)
        elif kernel == 'additive':
            term_1_vec = tf.map_fn(fn=lambda k: term_1_func_additive(net, mu, inputs, stddev, 
                                                                     scale, scale_mat, k_wz, Sigma_u,
                                                                     inputs_int[k], inputs_int[k+1], k_inv_k_wz),
                                       elems=tf.range(tf.cast(n_bags, dtype=tf.int32)),
                                       dtype=dtype)
        #term_1_vec = tf.Print(term_1_vec, [term_1_vec], '1')
        # We do not do multiple outputs, instead we recompute diag, as multiple outputs is CPU only...
        term_1 = tf.reduce_sum(tf.multiply(term_1_vec, inputs['y']))
        # sum mu^2
        mu_square = tf.multiply(mu, mu)
        # diag is transpose first one, elementwise multiply, sum across rows axis=0
        term_1_diag = tf.reduce_sum( tf.multiply(k_wz, k_inv_k_wz), axis=0) #diag K_zw K_ww^-1 k_wz
        k_zw_k_inv_S = tf.matmul(tf.transpose(k_inv_k_wz), Sigma_u) # k_zw K_ww^-1 Sigma_u
        term_2_diag = tf.reduce_sum(tf.multiply(tf.transpose(k_zw_k_inv_S), k_inv_k_wz), axis=0)
        # diagonal as [n_indiv]
        net.Sigma_diag = Sigma_diag = term_0_diag - term_1_diag + term_2_diag
        net.indiv = indiv = Sigma_diag + mu_square # E(X^2) is just normal second moment.
        term_2 = tf.reduce_sum(tf.multiply(indiv, inputs['indiv_pop']))
        # sum of all pop * (mu_square + sigma_diag)
        #indiv = tf.Print(indiv, [indiv, inputs['indiv_y']], message='indiv', summarize=5)

        #pop_mu = tf.multiply(inputs['indiv_pop'], tf.exp(mu))
        #pool_pop_mu = tf.squeeze(net.bag_pool(tf.expand_dims(pop_mu, 1))) #[n_bags]
        #term_1 = tf.reduce_sum(tf.multiply(inputs['y'], tf.log(pool_pop_mu)))

        # Term 2 \sum \sum p^i_j exp(\mu^i_j + Sigma^i_j/2)
        #pop_mu_sig = tf.multiply(inputs['indiv_pop'], tf.exp(mu + 0.5 * Sigma_diag))
        #term_2 = tf.reduce_sum(pop_mu_sig)

        # Term 3
        tfd = tf.contrib.distributions
        mvn_q = tfd.MultivariateNormalTriL(loc=tf.squeeze(params['mean']), scale_tril=triangular)
        mvn_u = tfd.MultivariateNormalTriL(loc=tf.tile(params['prior_mean'], [land_size]), scale_tril=chol_k)
        term_3 = tf.distributions.kl_divergence(mvn_q, mvn_u)
        
        #term_1 = tf.Print(term_1, [term_1/n_bags], message='1')
        #term_2 = tf.Print(term_2, [term_2/n_bags], message='2')
        #term_3 = tf.Print(term_3, [term_3/total_size], message='3')

        # Stirlings approximation to enable comparison across losses (\sum log (y_j !))
        zeros = tf.zeros_like(inputs['y']) # create a tensor all ones
        mask = tf.greater(inputs['y'], zeros) # boolean tensor, mask[i] = True iff x[i] > 1
        non_zero_y = tf.boolean_mask(inputs['y'], mask)
        #non_zero_y = tf.Print(non_zero_y, [non_zero_y, inputs['y']], summarize=100)
        term_4 = tf.reduce_sum(tf.multiply(non_zero_y, tf.log(non_zero_y)) - non_zero_y + 0.5 * tf.log(2.0 * pi * non_zero_y))
        #term_4 = tf.Print(term_4, [term_4/n_bags], message='4')
        
        net.loss  = -1.0/n_bags * (term_1 - term_2 - term_4) + term_3/total_size

        #if MAP:
        #net.indiv = indiv = tf.exp(mu - Sigma_diag)
        #else:
        net.indiv_se = net.square_err(inputs['indiv_true_y'], indiv)
        net.indiv_nll = net.nll_term(inputs['indiv_y'], indiv)

        #indiv = tf.Print(indiv, [indiv], summarize =200, message='indiv')
        #indiv_mean = tf.exp(mu + 0.5 * Sigma_diag)
        net.indiv_y = indiv_y_pop = tf.multiply(inputs['indiv_pop'], indiv)
        indiv_y_pop = tf.expand_dims(indiv_y_pop, 1)
        net.bag_y = bag_y = tf.squeeze(net.bag_pool(indiv_y_pop))
        #bag_y = tf.Print(bag_y, [bag_y, inputs['y']], message='bag', summarize=5)
        net.bag_se = net.square_err(inputs['y'], bag_y, bags=True)
        net.bag_nll = net.nll_term(inputs['y'], bag_y, bags=True)

        #indiv_y_mean = tf.multiply(inputs['indiv_pop'], tf.exp(mu + 0.5 * Sigma_diag))
        #indiv_y_var = tf.multiply(tf.exp(Sigma_diag) - 1.0, tf.exp( 2.0* mu + Sigma_diag) )
        #indiv_y = tf.Print(indiv_y, [indiv_y_mean, inputs['indiv_y'], indiv_y_var], summarize=2)
        #net.bag_se = tf.reduce_sum(tf.square(bag_y - inputs['y']))
        #if indiv_y_bol:
        #    net.indiv_se = tf.reduce_sum(tf.square(indiv_y - inputs['indiv_y']))
        # Can add net.print_out
    return net


