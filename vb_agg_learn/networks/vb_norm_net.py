# Variational optimisation- Normal 
from __future__ import division, print_function
from functools import partial

import numpy as np
import tensorflow as tf
from math import log, pi

from .base import Network
from vb_utility import make_stable, triangular_vec, fill_triangular

def build_net(in_dim, n_hidden, data_type, link='exp', total_size=None, var_init=0.01,
              bw_indiv=1.0, indiv_y_bol=False, kernel='ard', initialse='identity',
              seed=23, dtype=tf.float32, landmarks=None, avg_label=1.0, **others):
    #print('avg_label', avg_label)
    net = Network(in_dim, data_type, n_hidden=n_hidden, link=link, kernel=kernel, var_init=var_init,
                  indiv_bol=indiv_y_bol, dtype=dtype, seed=seed)
    inputs = net.inputs
    params = net.params
    land_size = n_hidden

    cst = partial(tf.cast, dtype=dtype)
    # Model parameters
    initializer = tf.initializers.random_normal(seed=seed, dtype=dtype) # normal initialiser 
    z_initializer = tf.zeros_initializer(dtype=dtype)
    o_initializer = tf.ones_initializer(dtype=dtype)
    #initializer = tf.keras.initializers.he_normal(seed=seed)
    print('bw_indiv', bw_indiv)
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
    # Intialise with L = I for safe inversion at start.
    params['L'] = tf.Variable(triangle_vec, name= 'L', dtype=dtype)
    params['mean'] = tf.Variable(avg_label * o_initializer([land_size, 1]), name = 'mean', dtype=dtype)
    #tf.Variable(tf.tile(tf.constant([7.0], dtype=dtype), land_size))
    #tf.Variable(z_initializer([land_size, 1]), name = 'mean', dtype=dtype)
    params['prior_mean'] =  tf.Variable(z_initializer([1]), name = 'prior_mean', dtype=dtype)
    #tf.Variable(tf.constant([7.0], dtype=dtype), name = 'prior_mean', dtype=dtype)
    #tf.Variable(initializer([1]), name = 'prior_mean', dtype=dtype)
    
    if kernel in ['ard', 'additive']:
        params['log_bw_sq'] = tf.Variable(tf.log(tf.square(tf.constant(bw_indiv, dtype=dtype))), name = 'log_bw_sq')
        #params['log_bw_sq'] = tf.log(tf.square(tf.constant(bw_indiv, dtype=dtype)), name = 'log_bw_sq')
    elif kernel == 'rbf':
        print('Vary Bandwidth RBF')
        params['log_bw_sq'] = tf.Variable(tf.log(tf.square(tf.constant(bw_indiv, dtype=dtype))), name = 'log_bw_sq')
        #params['log_bw_sq'] = tf.log(tf.square(tf.constant(bw_indiv, dtype=dtype)))

    n_bags = cst(tf.shape(inputs['sizes'])[0])
    n_indiv = cst(tf.shape(inputs['X'])[0])

    sigma_sq = tf.exp(params['log_sig_sq'])
    scale = tf.exp(params['log_scale'])
    stddev = tf.sqrt(tf.exp(params['log_bw_sq']))
    #stddev = tf.Print(stddev, [stddev], message='bw', summarize=18)
    landmarks = inputs['landmarks']
    inputs_int = tf.concat([tf.constant([0], tf.int32), tf.cumsum(tf.cast(inputs['sizes'], tf.int32))], 0)
    #inputs_int = tf.Print(inputs_int, [inputs_int])

    Sigma_term0 = tf.map_fn(fn=lambda k:
                            tf.reduce_sum(scale * net.kernel(inputs['X'][inputs_int[k]:inputs_int[k+1],:], 
                                   inputs['X'][inputs_int[k]:inputs_int[k+1],:], 
                                   stddev=stddev, scale=1.0)),
                                   elems=tf.range(tf.cast(n_bags, dtype=tf.int32)),
                                   dtype=dtype)

    if kernel in ['ard', 'rbf']:
        k_ww = scale * net.kernel(landmarks, landmarks, stddev=stddev, scale=1.0)
        k_wz = scale * net.kernel(landmarks, inputs['X'], stddev=stddev, scale=1.0) #K_wz
        k_zz = scale * net.kernel(inputs['X'], inputs['X'], stddev=stddev, scale=1.0) #Change k_zz
        #k_ww = tf.Print(k_ww, [k_ww], message='k_ww', summarize=100)
        #k_wz = tf.Print(k_wz, [k_wz], message='k_wz', summarize=100)
        #k_zz = tf.Print(k_zz, [k_zz], message='k_zz', summarize=100)

        #k_wz = tf.Print(k_wz, [k_wz])
        term_0_diag = scale * tf.ones([tf.cast(n_indiv, dtype=tf.int32)], dtype=dtype) #k_zz diagonal
    elif kernel == 'additive':
        scale_mat = tf.exp(params['log_scale_m'])
        k_ww = net.kernel(landmarks, landmarks, stddev_ard=stddev[:-2], scale_ard=scale, 
                                                stddev_mat=stddev[-2:], scale_mat=scale_mat)
        k_wz = net.kernel(landmarks, inputs['X'], stddev_ard=stddev[:-2], scale_ard=scale, 
                                                  stddev_mat=stddev[-2:], scale_mat=scale_mat)
        #term_0_diag = (scale + scale_mat) * tf.ones([tf.cast(n_indiv, dtype=tf.int32)], dtype=dtype) #k_zz diagonal
    # SLOW: Compute full kernel matrix and then pool pool then take diag.
    #Sigma_term0 = tf.diag_part(net.bag_pool(tf.transpose(net.bag_pool(k_zz))))
    #Sigma_term0 = tf.Print(Sigma_term0, [Sigma_term0, net.bag_pool(k_zz), net.bag_pool(tf.transpose(net.bag_pool(k_zz)))], message='Sigma0', summarize=1000)
    #Sigma_term0 = tf.Print(Sigma_term0, [Sigma_term0, batch_items], summarize=100)
    chol_k = tf.cholesky(k_ww)
    k_ww_inv = tf.matrix_inverse(k_ww) # K_ww^-1
    triangular = fill_triangular(params['L']) #\Sigma_u=LL^T

    Sigma_u = tf.matmul(triangular, tf.transpose(triangular)) # Sigma_u = L L^T
    
    pool_kzw = net.bag_pool(tf.transpose(k_wz))
    #pool_kzw = tf.Print(pool_kzw, [tf.transpose(k_wz), pool_kzw], message='pool_kzw', summarize=100)
    pool_k_zw_k_ww_inv = tf.matmul(pool_kzw, k_ww_inv)
    kw_zw_k_ww_inv = tf.matmul(tf.transpose(k_wz), k_ww_inv)
    #pool_k_zw_k_ww_inv = tf.Print(pool_k_zw_k_ww_inv, [tf.matmul(tf.matmul(tf.transpose(k_wz), k_ww_inv), k_wz)], summarize=100, message='sum')
    #Sigma_term1_check = tf.diag_part(tf.matmul(pool_k_zw_k_ww_inv, tf.transpose(pool_kzw)))
    # Check this: transpose latter and elementwise multiply, sum across axis=1
    Sigma_term1 = tf.reduce_sum(tf.multiply(pool_k_zw_k_ww_inv, pool_kzw), axis=1)
    #Sigma_term1 = tf.Print(Sigma_term1, [Sigma_term1, Sigma_term1_check], message='Sigma_term1')

    pool_k_zw_k_ww_inv_Sig_u = tf.matmul(pool_k_zw_k_ww_inv, Sigma_u)
    #pool_k_zw_k_ww_inv = tf.Print(pool_k_zw_k_ww_inv, [tf.matmul(tf.matmul(kw_zw_k_ww_inv, Sigma_u), tf.transpose(kw_zw_k_ww_inv))], summarize=100, message='sum_2')

    #Sigma_term2_check = tf.diag_part(tf.matmul(pool_k_zw_k_ww_inv_Sig_u, tf.transpose(pool_k_zw_k_ww_inv)))
    # Check this: transpose latter and elementwise multiply, sum across axis=1
    Sigma_term2 = tf.reduce_sum(tf.multiply(pool_k_zw_k_ww_inv_Sig_u, pool_k_zw_k_ww_inv), axis=1)
    #Sigma_term2 = tf.Print(Sigma_term2, [Sigma_term2, Sigma_term2_check], message='Sigma_term2')

    Sigma_sum_term = Sigma_term0 - Sigma_term1 + Sigma_term2 
    #Sigma_sum_term = tf.Print(Sigma_sum_term, [Sigma_term0, Sigma_term1, Sigma_term2])
    k_inv_k_wz = tf.matmul(k_ww_inv, k_wz) # K_ww^-1 K_wz
    mean_diff = params['mean'] - params['prior_mean']
    #mean_diff = tf.Print(mean_diff, [tf.shape(mean_diff)], message='mean_diff')
    net.mu = mu = params['prior_mean'] + tf.squeeze(tf.matmul(tf.transpose(k_inv_k_wz), mean_diff)) 
    # mu_prior + K_zw K_ww^-1 (mu_u - mu_prior)
    mu_pool = tf.squeeze(net.bag_pool(tf.expand_dims(mu, 1))) # 1^T mu [bags]
    #mu_pool = tf.Print(mu_pool, [mu_pool, mu], message='mu_pool')

    term_1_0 = tf.square(inputs['y']) #sum_j y_j^2
    term_1_1 = 2.0 * tf.multiply(inputs['y'], mu_pool) # 2 * sum_j(y_j *1^T mu)
    term_1_2 = Sigma_sum_term # 1^T S 1
    term_1_3 = tf.square(mu_pool) # \sum_j 1^T mu_j mu_j^t 1 = \sum_j (mu_j^t 1)^2

    # Term 1
    #sigma_sq = tf.Print(sigma_sq, [sigma_sq], 'sigma^2')
    bag_sigma_sq = sigma_sq*inputs['sizes']
    #bag_sigma_sq = tf.Print(bag_sigma_sq, [bag_sigma_sq, term_1_0, term_1_1, term_1_2, term_1_3], message='bag_sigma_sq')
    term_1_rescale = tf.divide(term_1_0 - term_1_1 + term_1_2 + term_1_3, bag_sigma_sq)
    term_1 = tf.reduce_sum(term_1_rescale)
    
    # Term 2 \sum_j log(2 pi sigma^2_j)
    term_2 = tf.reduce_sum(tf.log(2.0 * pi * bag_sigma_sq))
    
    # Term 3
    tfd = tf.contrib.distributions
    mvn_q = tfd.MultivariateNormalTriL(loc=tf.squeeze(params['mean']), scale_tril=triangular)
    mvn_u = tfd.MultivariateNormalTriL(loc=tf.tile(params['prior_mean'], [land_size]), scale_tril=chol_k)
    term_3 = tf.distributions.kl_divergence(mvn_q, mvn_u)
    #term_3 = tf.Print(term_3, [0.5* term_1/n_bags, 0.5* term_2/n_bags, term_3/total_size], message='all_terms')
    term_1_diag = tf.reduce_sum( tf.multiply(k_wz, k_inv_k_wz), axis=0) #diag K_zw K_ww^-1 k_wz
    #term_1_diag_check = tf.diag_part(tf.matmul(tf.transpose(k_wz), k_inv_k_wz))
    k_zw_k_inv_S = tf.matmul(tf.transpose(k_inv_k_wz), Sigma_u) # k_zw K_ww^-1 Sigma_u
    #term_2_diag_check = tf.diag_part(tf.matmul(k_zw_k_inv_S, k_inv_k_wz))
    term_2_diag = tf.reduce_sum(tf.multiply(tf.transpose(k_zw_k_inv_S), k_inv_k_wz), axis=0)
    # diagonal as [n_indiv]
    #Sigma_diag_check = Sigma_diag = term_0_diag - term_1_diag + term_2_diag
    net.Sigma_diag = Sigma_diag = term_0_diag - term_1_diag + term_2_diag
    #term_1 = tf.Print(term_1, [term_0_diag,term_1_diag, term_2_diag, tf.sqrt(Sigma_diag), tf.sqrt(Sigma_diag_check)], summarize=3, message='Sigma_diag')
    net.loss = loss = -1.0/n_bags * (-0.5*term_1 - 0.5*term_2) + term_3/total_size

    #if MAP:
    #net.indiv = indiv = tf.exp(mu - Sigma_diag)
    #else:


    net.indiv = indiv = mu #tf.squeeze(mu + 0.5 * Sigma_diag))
    #indiv = tf.Print(indiv, [indiv], message='mu', summarize=5)
    
    #net.indiv = indiv = tf.exp(mu - Sigma_diag)
    net.indiv_se = net.square_err(inputs['indiv_true_y'], indiv)
    net.indiv_nll = net.nll_term(inputs['indiv_y'], indiv)

    #indiv = tf.Print(indiv, [indiv], summarize =200, message='indiv')
    #indiv_mean = tf.exp(mu + 0.5 * Sigma_diag)
    net.indiv_y = indiv_y_pop = tf.multiply(inputs['indiv_pop'], indiv)
    indiv_y_pop = tf.expand_dims(indiv_y_pop, 1)
    net.bag_y = bag_y = tf.squeeze(net.bag_pool(indiv_y_pop))
    bag_y = tf.Print(bag_y, [bag_y, inputs['y']], message='bag')
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
