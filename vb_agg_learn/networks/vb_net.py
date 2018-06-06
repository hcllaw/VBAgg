# Variational optimisation
from __future__ import division, print_function
from functools import partial

import numpy as np
import tensorflow as tf
from math import log, pi

from .base import Network
from vb_utility import make_stable, triangular_vec, fill_triangular

def build_net(in_dim, n_hidden, data_type, link='exp', total_size=None,
              bw_indiv=1.0, indiv_y_bol=False, kernel='ard', initialse='identity',
              seed=23, dtype=tf.float32, landmarks=None, log_y=False,  device_name=None,
              avg_label=1.0, **others):
    with tf.device(device_name):
        net = Network(in_dim, data_type, n_hidden=n_hidden, link=link, kernel=kernel,
                      indiv_bol=indiv_y_bol, dtype=dtype, seed=seed, log_y=log_y)
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
                init_kernel = net.kernel(landmarks, landmarks, stddev_ard=bw_indiv[:-2], scale_ard=0.5, 
                                                               stddev_mat=bw_indiv[-2:], scale_mat=0.5, 
                                                               tensorf=False)
            elif kernel in ['rbf', 'ard']:
                init_kernel = net.kernel(landmarks, landmarks, stddev=bw_indiv, scale=1.0, tensorf=False)
            L = np.linalg.cholesky(init_kernel)
            triangle_vec = tf.constant(triangular_vec(L, n=land_size), dtype=dtype)
        #print('bw_indiv', bw_indiv)
        mean_scale = log(avg_label) - 0.5 # Predict Baseline
        #print('mean_scale', mean_scale)
        params['L'] = tf.Variable(triangle_vec, name= 'L', dtype=dtype)
        params['mean'] = tf.Variable(mean_scale * o_initializer([land_size, 1]), name = 'mean', dtype=dtype)
        params['prior_mean'] = tf.Variable(z_initializer([1]), name = 'prior_mean', dtype=dtype)

        # Change to z_initializer for malaria
        if kernel in ['ard', 'additive']:
            params['log_bw'] = tf.Variable(tf.log(tf.constant(bw_indiv, dtype=dtype)), name = 'log_bw')
        elif kernel == 'rbf':
            #print('Vary Bandwidth RBF')
            params['log_bw'] = tf.Variable(tf.log(tf.constant(bw_indiv, dtype=dtype)), name = 'log_bw')

        n_bags = cst(tf.shape(inputs['sizes'])[0])
        n_indiv = cst(tf.shape(inputs['X'])[0])

        scale = tf.exp(params['log_scale'])
        stddev = tf.exp(params['log_bw'])
        
        landmarks = inputs['landmarks']
        #stddev = tf.Print(stddev, [stddev], message='bw', summarize=10000)
        if kernel in ['ard', 'rbf']:
            k_ww = scale * net.kernel(landmarks, landmarks, stddev=stddev, scale=1.0)
            k_wz = scale * net.kernel(landmarks, inputs['X'], stddev=stddev, scale=1.0) #K_wz
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
        k_ww_inv = tf.check_numerics(k_ww_inv, 'k_ww_inv')

        triangular = fill_triangular(params['L']) #\Sigma_u=LL^T

        #triangular = tf.matrix_set_diag(triangular, tf.exp(tf.matrix_diag_part(triangular)))
        Sigma_u = tf.matmul(triangular, tf.transpose(triangular)) # Sigma_u = L L^T

        k_inv_k_wz = tf.matmul(k_ww_inv, k_wz) # K_ww^-1 K_wz
        mean_diff = params['mean'] - params['prior_mean']

        net.mu = mu = params['prior_mean'] + tf.squeeze(tf.matmul(tf.transpose(k_inv_k_wz), mean_diff)) # mu_prior + K_zw K_ww^-1 (mu_u - mu_prior)
        mu = tf.check_numerics(mu, 'mu')
        # diag is transpose first one, elementwise multiply, sum across rows axis=0
        term_1_diag = tf.reduce_sum( tf.multiply(k_wz, k_inv_k_wz), axis=0) #diag K_zw K_ww^-1 k_wz
        term_1_diag = tf.check_numerics(term_1_diag, 'term_1_diag')
        #term_1_diag_check = tf.diag_part(tf.matmul(tf.transpose(k_wz), k_inv_k_wz))
        k_zw_k_inv_S = tf.matmul(tf.transpose(k_inv_k_wz), Sigma_u) # k_zw K_ww^-1 Sigma_u
        k_zw_k_inv_S = tf.check_numerics(k_zw_k_inv_S, 'k_zw_k_inv_S')
        # diag k_zw K_ww^-1 S K_ww^-1 k_wz (tranpose first one, hence sum rows axis=0)
        term_2_diag = tf.reduce_sum(tf.multiply(tf.transpose(k_zw_k_inv_S), k_inv_k_wz), axis=0)
        term_2_diag = tf.check_numerics(term_2_diag, 'term_2_diag')
        #term_2_diag_check = tf.diag_part(tf.matmul(k_zw_k_inv_S, k_inv_k_wz))

        # diagonal as [n_indiv]
        net.Sigma_diag = Sigma_diag = term_0_diag - term_1_diag + term_2_diag
        net.Sigma_diag = tf.check_numerics(net.Sigma_diag, 'Sigma_diag')

        # Term 1 \sum y_j log (\sum p^i_j exp mu^i_j)
        pop_mu = tf.multiply(inputs['indiv_pop'], tf.exp(mu))
        pool_pop_mu = tf.squeeze(net.bag_pool(tf.expand_dims(pop_mu, 1))) #[n_bags]
        term_1 = tf.reduce_sum(tf.multiply(inputs['y'], tf.log(pool_pop_mu)))
        term_1 = tf.check_numerics(term_1, 'term_1')

        # Term 2 \sum \sum p^i_j exp(\mu^i_j + Sigma^i_j/2)
        pop_mu_sig = tf.multiply(inputs['indiv_pop'], tf.exp(mu + 0.5 * Sigma_diag))
        term_2 = tf.reduce_sum(pop_mu_sig)
        term_2 = tf.check_numerics(term_2, 'term_2')

        # Term 3
        tfd = tf.contrib.distributions
        mvn_q = tfd.MultivariateNormalTriL(loc=tf.squeeze(params['mean']), scale_tril=triangular)
        mvn_u = tfd.MultivariateNormalTriL(loc=tf.tile(params['prior_mean'], [land_size]), scale_tril=chol_k)
        term_3 = tf.distributions.kl_divergence(mvn_q, mvn_u)
        # Tr(K_ww^-1 Sigma_u) = sum elementwise A * B.T 
        #term_3_1 = tf.reduce_sum(tf.multiply(k_ww_inv, Sigma_u))
        #term_3_1_check = tf.trace(tf.matmul(K_ww_inv, Sigma_u))
        #term_3_1 = tf.Print(term_3_1, [term_3_1, term], message='')
        # log (det(K_ww)/det(Sigma_u)), det(Sigma_u) = det(L) * det(L^T), 
        # det(L)=product of diagonal https://proofwiki.org/wiki/Determinant_of_Triangular_Matrix
        # numerical stability log (\product diag(L))**2 =  2.0 * \sum log (diag(L)_i) NOTE: order!
        # We use abs here to make the value positive, but maybe exp better, but worry about learning rate updates.
        #log_det_Sigma_u = 2.0 * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(triangular))))
        #log_det_Sigma_u = tf.Print(log_det_Sigma_u, [log_det_Sigma_u, tf.log(det_k_ww)], message='log_det_Sigma_u')

        #term_3_2 = log_det_k_ww - log_det_Sigma_u # care that determinant near 0, intialise with +e-5?
        # Assume mu_w=0, i.e. mu_u^T K_ww^-1 mu_u
        #term_3_3 = tf.matmul(tf.matmul(tf.transpose(mean_diff), k_ww_inv), mean_diff)
        #term_3_3 = tf.squeeze(term_3_3)
        #term_3 = 0.5 * (term_3_1 + term_3_2 + term_3_3 - land_size)
        #term_3 = tf.Print(term_3, [term_3, term_3_check], message='kl_check')
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
        net.indiv = indiv = tf.squeeze(tf.exp(mu + 0.5 * Sigma_diag))
        #indiv = tf.Print(indiv, [indiv])
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
