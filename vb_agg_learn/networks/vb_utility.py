# VAE utilities
from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import print_function

import functools
import hashlib

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

import scipy.stats as ss
import numpy as np
import tensorflow as tf

def term_1_func(net, mu, inputs, stddev, scale, k_wz, Sigma_u, lower, upper, k_inv_k_wz):
    #mu = tf.Print(mu, [mu], 'mu')
    sqrt_pop = tf.sqrt(inputs['indiv_pop'][lower:upper])
    sqrt_pop_matrix = tf.matmul(tf.expand_dims(sqrt_pop, 1), tf.expand_dims(sqrt_pop, 0))
    #sqrt_pop_matrix = tf.Print(sqrt_pop_matrix, [sqrt_pop_matrix], 'pop_matrix')
    mu_j = tf.multiply(sqrt_pop, mu[lower:upper])
    #mu_j = tf.Print(mu_j, [mu_j], 'mu_j')
    k_zz = net.kernel(inputs['X'][lower:upper,:], 
                      inputs['X'][lower:upper,:], 
                      stddev=stddev, scale=scale)
    #k_zz = tf.Print(k_zz, [tf.shape(k_zz)], 'k_zz')
    k_zw = tf.transpose(k_wz)[lower:upper, :]
    #k_zw = tf.Print(k_zw, [tf.shape(k_zw)], 'k_zw')
    # norm square of mu
    norm = tf.square(tf.norm(mu_j, ord=2))
    #norm = tf.Print(norm, [norm], 'norm')
    # k_zw k_ww^-1
    k_zw_k_ww_inv = tf.transpose(k_inv_k_wz)[lower:upper, :]
    #k_zw_k_ww_inv = tf.Print(k_zw_k_ww_inv, [tf.shape(k_zw_k_ww_inv)], 'k_zw_k_ww_inv')
    # k_zw k_ww^-1 k_wz
    k_zw_k_ww_inv_k_wz = tf.matmul(k_zw_k_ww_inv, tf.transpose(k_zw))
    #k_zw_k_ww_inv_k_wz = tf.Print(k_zw_k_ww_inv_k_wz, [tf.shape(k_zw_k_ww_inv_k_wz)], 'k_zw_k_ww_inv_k_wz')
    # k_zw k_ww^-1 Sigma_u k_ww^-1 k_wz
    k_zw_k_ww_inv_S = tf.matmul(tf.matmul(k_zw_k_ww_inv, Sigma_u), tf.transpose(k_zw_k_ww_inv))
    #k_zw_k_ww_inv_S = tf.Print(k_zw_k_ww_inv_S, [tf.shape(k_zw_k_ww_inv_S)], 'k_zw_k_ww_inv_S')
    # Sigma_j = k_zz - k_zw k_ww^-1 k_wz + k_zw k_ww^-1 Sigma_u k_ww^-1 k_wz
    Sigma_j = k_zz - k_zw_k_ww_inv_k_wz + k_zw_k_ww_inv_S
    #Sigma_j = tf.Print(Sigma_j, [tf.shape(Sigma_j)], 'Sigma_j')
    Sigma_j = tf.multiply(Sigma_j, sqrt_pop_matrix)
    #Sigma_j = tf.Print(Sigma_j, [tf.shape(Sigma_j), tf.trace(Sigma_j)], 'Sigma_j')
    # norm square of mu_j + tr(Sigma_j)

    norm_Sigma = norm + tf.trace(Sigma_j)
    term_1_0 = tf.log(norm_Sigma)
    #term_1_0 = tf.Print(term_1_0, [term_1_0], 'term_1_0')
    # mu^T Sigma mu
    mu_j = tf.expand_dims(mu_j, 1)
    term_1_0_upper = 2.0 * tf.squeeze(tf.matmul(tf.matmul(tf.transpose(mu_j), Sigma_j), mu_j))
    #term_1_0_upper = tf.Print(term_1_0_upper, [tf.shape(term_1_0_upper)], 'term_1_0_upper')
    trace_sigma_sq = tf.reduce_sum(tf.multiply(Sigma_j, Sigma_j))
    #trace_sigma_sq_check = tf.trace(tf.matmul(Sigma_j, Sigma_j))
    #trace_sigma_sq = tf.Print(trace_sigma_sq, [trace_sigma_sq, trace_sigma_sq_check])
    term_1_1 = tf.add(term_1_0_upper, trace_sigma_sq) / tf.square(norm_Sigma)
    #term_1_1 = tf.Print(term_1_1, [term_1_1], 'term_1_1')
    return term_1_0 - term_1_1

def term_1_func_additive(net, mu, inputs, stddev, scale_ard, scale_mat, 
                         k_wz, Sigma_u, lower, upper, k_inv_k_wz):
    stddev_ard = stddev[:-2]
    stddev_mat = stddev[-2:]
    #mu = tf.Print(mu, [mu], 'mu')
    sqrt_pop = tf.sqrt(inputs['indiv_pop'][lower:upper])
    sqrt_pop_matrix = tf.matmul(tf.expand_dims(sqrt_pop, 1), tf.expand_dims(sqrt_pop, 0))
    #sqrt_pop_matrix = tf.Print(sqrt_pop_matrix, [sqrt_pop_matrix], 'pop_matrix')
    mu_j = tf.multiply(sqrt_pop, mu[lower:upper])
    #mu_j = tf.Print(mu_j, [mu_j], 'mu_j')
    k_zz = net.kernel(inputs['X'][lower:upper,:], 
                      inputs['X'][lower:upper,:], 
                      stddev_ard=stddev_ard, scale_ard=scale_ard, 
                      stddev_mat=stddev_mat, scale_mat=scale_mat)
    #k_zz = tf.Print(k_zz, [tf.shape(k_zz)], 'k_zz')
    k_zw = tf.transpose(k_wz)[lower:upper, :]
    #k_zw = tf.Print(k_zw, [tf.shape(k_zw)], 'k_zw')
    # norm square of mu
    norm = tf.square(tf.norm(mu_j, ord=2))
    #norm = tf.Print(norm, [norm], 'norm')
    # k_zw k_ww^-1
    k_zw_k_ww_inv = tf.transpose(k_inv_k_wz)[lower:upper, :]
    #k_zw_k_ww_inv = tf.Print(k_zw_k_ww_inv, [tf.shape(k_zw_k_ww_inv)], 'k_zw_k_ww_inv')
    # k_zw k_ww^-1 k_wz
    k_zw_k_ww_inv_k_wz = tf.matmul(k_zw_k_ww_inv, tf.transpose(k_zw))
    #k_zw_k_ww_inv_k_wz = tf.Print(k_zw_k_ww_inv_k_wz, [tf.shape(k_zw_k_ww_inv_k_wz)], 'k_zw_k_ww_inv_k_wz')
    # k_zw k_ww^-1 Sigma_u k_ww^-1 k_wz
    k_zw_k_ww_inv_S = tf.matmul(tf.matmul(k_zw_k_ww_inv, Sigma_u), tf.transpose(k_zw_k_ww_inv))
    #k_zw_k_ww_inv_S = tf.Print(k_zw_k_ww_inv_S, [tf.shape(k_zw_k_ww_inv_S)], 'k_zw_k_ww_inv_S')
    # Sigma_j = k_zz - k_zw k_ww^-1 k_wz + k_zw k_ww^-1 Sigma_u k_ww^-1 k_wz
    Sigma_j = k_zz - k_zw_k_ww_inv_k_wz + k_zw_k_ww_inv_S
    #Sigma_j = tf.Print(Sigma_j, [tf.shape(Sigma_j)], 'Sigma_j')
    Sigma_j = tf.multiply(Sigma_j, sqrt_pop_matrix)
    #Sigma_j = tf.Print(Sigma_j, [tf.shape(Sigma_j), tf.trace(Sigma_j)], 'Sigma_j')
    # norm square of mu_j + tr(Sigma_j)

    norm_Sigma = norm + tf.trace(Sigma_j)
    term_1_0 = tf.log(norm_Sigma)
    #term_1_0 = tf.Print(term_1_0, [term_1_0], 'term_1_0')
    # mu^T Sigma mu
    mu_j = tf.expand_dims(mu_j, 1)
    term_1_0_upper = 2.0 * tf.squeeze(tf.matmul(tf.matmul(tf.transpose(mu_j), Sigma_j), mu_j))
    #term_1_0_upper = tf.Print(term_1_0_upper, [tf.shape(term_1_0_upper)], 'term_1_0_upper')
    trace_sigma_sq = tf.reduce_sum(tf.multiply(Sigma_j, Sigma_j))
    #trace_sigma_sq_check = tf.trace(tf.matmul(Sigma_j, Sigma_j))
    #trace_sigma_sq = tf.Print(trace_sigma_sq, [trace_sigma_sq, trace_sigma_sq_check])
    term_1_1 = tf.add(term_1_0_upper, trace_sigma_sq) / tf.square(norm_Sigma)
    #term_1_1 = tf.Print(term_1_1, [term_1_1], 'term_1_1')
    return term_1_0 - term_1_1

def fill_triangular(x, upper=False, name=None):
    """Creates a (batch of) triangular matrix from a vector of inputs.
    Created matrix can be lower- or upper-triangular. (It is more efficient to
    create the matrix as upper or lower, rather than transpose.)
    Triangular matrix elements are filled in a clockwise spiral. See example,
    below.
    If `x.get_shape()` is `[b1, b2, ..., bK, d]` then the output shape is `[b1,
    b2, ..., bK, n, n]` where `n` is such that `d = n(n+1)/2`, i.e.,
    `n = int(np.sqrt(0.25 + 2. * m) - 0.5)`.
    Example:
    ```python
    fill_triangular([1, 2, 3, 4, 5, 6])
    # ==> [[4, 0, 0],
    #      [6, 5, 0],
    #      [3, 2, 1]]
    fill_triangular([1, 2, 3, 4, 5, 6], upper=True)
    # ==> [[1, 2, 3],
    #      [0, 5, 6],
    #      [0, 0, 4]]
    ```
    For comparison, a pure numpy version of this function can be found in
    `util_test.py`, function `_fill_triangular`.
    Args:
      x: `Tensor` representing lower (or upper) triangular elements.
      upper: Python `bool` representing whether output matrix should be upper
        triangular (`True`) or lower triangular (`False`, default).
      name: Python `str`. The name to give this op.
    Returns:
      tril: `Tensor` with lower (or upper) triangular elements filled from `x`.
    Raises:
      ValueError: if `x` cannot be mapped to a triangular matrix.
    """

    with ops.name_scope(name, "fill_triangular", values=[x]):
      x = ops.convert_to_tensor(x, name="x")
      if x.shape.with_rank_at_least(1)[-1].value is not None:
        # Formula derived by solving for n: m = n(n+1)/2.
        m = np.int32(x.shape[-1].value)
        n = np.sqrt(0.25 + 2. * m) - 0.5
        if n != np.floor(n):
          raise ValueError("Input right-most shape ({}) does not "
                           "correspond to a triangular matrix.".format(m))
        n = np.int32(n)
        static_final_shape = x.shape[:-1].concatenate([n, n])
      else:
        m = array_ops.shape(x)[-1]
        # For derivation, see above. Casting automatically lops off the 0.5, so we
        # omit it.  We don't validate n is an integer because this has
        # graph-execution cost; an error will be thrown from the reshape, below.
        n = math_ops.cast(
            math_ops.sqrt(0.25 + math_ops.cast(2 * m, dtype=dtypes.float32)),
            dtype=dtypes.int32)
        static_final_shape = x.shape.with_rank_at_least(1)[:-1].concatenate(
            [None, None])
      # We now concatenate the "tail" of `x` to `x` (and reverse one of them).
      #
      # We do this based on the insight that the input `x` provides `ceil(n/2)`
      # rows of an `n x n` matrix, some of which will get zeroed out being on the
      # wrong side of the diagonal. The first row will not get zeroed out at all,
      # and we need `floor(n/2)` more rows, so the first is what we omit from
      # `x_tail`. If we then stack those `ceil(n/2)` rows with the `floor(n/2)`
      # rows provided by a reversed tail, it is exactly the other set of elements
      # of the reversed tail which will be zeroed out for being on the wrong side
      # of the diagonal further up/down the matrix. And, in doing-so, we've filled
      # the triangular matrix in a clock-wise spiral pattern. Neat!
      #
      # Try it out in numpy:
      #  n = 3
      #  x = np.arange(n * (n + 1) / 2)
      #  m = x.shape[0]
      #  n = np.int32(np.sqrt(.25 + 2 * m) - .5)
      #  x_tail = x[(m - (n**2 - m)):]
      #  np.concatenate([x_tail, x[::-1]], 0).reshape(n, n)  # lower
      #  # ==> array([[3, 4, 5],
      #               [5, 4, 3],
      #               [2, 1, 0]])
      #  np.concatenate([x, x_tail[::-1]], 0).reshape(n, n)  # upper
      #  # ==> array([[0, 1, 2],
      #               [3, 4, 5],
      #               [5, 4, 3]])
      #
      # Note that we can't simply do `x[..., -(n**2 - m):]` because this doesn't
      # correctly handle `m == n == 1`. Hence, we do nonnegative indexing.
      # Furthermore observe that:
      #   m - (n**2 - m)
      #   = n**2 / 2 + n / 2 - (n**2 - n**2 / 2 + n / 2)
      #   = 2 (n**2 / 2 + n / 2) - n**2
      #   = n**2 + n - n**2
      #   = n
      if upper:
        x_list = [x, array_ops.reverse(x[..., n:], axis=[-1])]
      else:
        x_list = [x[..., n:], array_ops.reverse(x, axis=[-1])]
      new_shape = (
          static_final_shape.as_list()
          if static_final_shape.is_fully_defined()
          else array_ops.concat([array_ops.shape(x)[:-1], [n, n]], axis=0))
      x = array_ops.reshape(array_ops.concat(x_list, axis=-1), new_shape)
      x = array_ops.matrix_band_part(
          x,
          num_lower=(0 if upper else -1),
          num_upper=(-1 if upper else 0))
      x.set_shape(static_final_shape)
      return x

# function to place 1 on diagonal when the output vector passes through tensorflow fill_triangular.
def triangular_vec(L, n=5):
    size = n * (n + 1) // 2 #int division
    x = np.arange(size)
    m = x.shape[0]
    x_tail = x[(m - (n**2 - m)):]
    triangular = np.concatenate([x_tail, x[::-1]], 0).reshape(n, n)
    vec = np.zeros(int(size))
    if L is not None:
      for i in range(n):
        vec[triangular[i:, i]] = L[i:, i]
    else:
      diag = np.diag(triangular)
      vec[diag] = 1
    vec.astype(np.float32)
    return vec

def triangular_fill_vec(n=5):
    size = n * (n + 1) // 2 #int division
    x = np.arange(size)
    m = x.shape[0]
    x_tail = x[(m - (n**2 - m)):]
    triangular = np.concatenate([x_tail, x[::-1]], 0).reshape(n, n)
    diag = np.diag(triangular)
    vec = np.zeros(int(size))
    vec[diag] = 1
    vec.astype(np.float32)
    return vec

def log_normal_modes(mu, sigma, pop, sizes, simulations=500):
    #print('mu', mu)
    #print('sigma', sigma)
    indexes = [0] + np.cumsum(sizes).tolist()
    n_bags = len(sizes)
    total_size = np.sum(sizes)
    snorm_sim = np.random.normal(size=(total_size, simulations))
    norm_sim = np.multiply(snorm_sim, np.expand_dims(sigma, 1)) + np.expand_dims(mu, 1)
    log_norm_sim = np.multiply(np.exp(norm_sim), np.expand_dims(pop, 1))
    bag_modes = np.zeros(n_bags)
    for i in range(n_bags):
        lower = indexes[i]
        upper = indexes[i+1]
        bag_modes[i] = mode_find(np.sum(log_norm_sim[lower:upper, :], axis=0)) 
    return bag_modes

# KDE or could bin...
def mode_find(values):
    kde_density = ss.gaussian_kde(values)
    x = np.linspace(np.min(values), np.max(values), 250)
    pdf_values = kde_density(x)
    return x[np.argsort(pdf_values)[-1]]

def make_stable(A, reg=1e-5, sqrt=False, inverse=False):
    reg = tf.constant([reg], dtype=A.dtype)
    # Note tensorflow SVD backprop may not be stable
    A = tf.check_numerics(A, 'A')
    #A = tf.Print(A, [A])
    #e, V = tf.self_adjoint_eig(A)
    #A_stable = tf.matmul(tf.matmul(V, tf.diag(e)), tf.matrix_inverse(V))
    #A_stable = tf.check_numerics(A_stable, 'A_stable')
    #A_stable = tf.Print(A_stable, [A_stable])
    si, u, v = tf.svd(A, full_matrices=True)
    si = tf.check_numerics(si, 'si')
    u = tf.check_numerics(u, 'u')
    v = tf.check_numerics(v, 'v')

    #si = tf.Print(si, [si], summarize=1000)
    #small_p_ev = tf.reduce_min(tf.boolean_mask(s, s > 0))
    #adjust = tf.minimum(small_p_ev, reg)
    #s = tf.Print(s, [s], message='s',summarize=100)
    #si = tf.where(tf.less_equal(s, 0), tf.tile(adjust, [tf.shape(s)[0]]), s)
    #si = tf.Print(si, [si], message='after',summarize=100)

    if sqrt:
        si = tf.sqrt(si)
    if inverse:
        si = tf.divide(1.0, si)
    #si = tf.Print(si, [si], summarize=1000)
    A_stable = tf.matmul(tf.matmul(u, tf.diag(si)), tf.transpose(v))
    return A_stable

def log_norm_quantile(mu, sigma, quantile=0.9, log=False):
    # Log normal quantile is just normal quantile exponentiated (CHECKED)
    lower, upper = ss.norm.interval(quantile)
    lower_quantile = mu + lower*sigma
    upper_quantile = mu + upper*sigma
    if log:
      return np.exp(lower_quantile), np.exp(upper_quantile)
    else:
      return lower_quantile, upper_quantile