from __future__ import print_function, division

import tensorflow as tf
import numpy as np
from scipy.stats import norm, t, uniform
from math import pi, sqrt
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.gaussian_process.kernels import Matern
# Class for random Fourier Features, matern32 + rbf
class RandomFourierFeatureMapper():
    def __init__(self, in_dim, n_rff=500, kernel='matern', seed=23):
        self.in_dim = in_dim
        self.n_rff = n_rff
        self.seed = seed
        self.kernel = kernel
        self.rs = check_random_state(self.seed)

    def sample(self, stddev=1.0, nu=1.5):
        if self.kernel == 'gaussian':
            generator = norm(loc=0.0, scale=1.0)
        elif self.kernel == 'matern':
            generator = t(df=2.0*nu)
        else:
            raise ValueError('{} kernel not implemented'.format(self.kernel))
        r = generator.rvs(size=(self.in_dim, self.n_rff), random_state=self.rs) / stddev
        return r 

    def map(self, inputs, stddev=1.0, nu=1.5, tf_version=True):
        dtype = inputs.dtype
        samples = self.sample(stddev, nu)
        bias = 2.0 * pi * uniform.rvs(size=self.n_rff, random_state=self.rs)
        if tf_version:
            bias = tf.constant(bias, dtype=dtype)
            samples = tf.constant(samples, dtype=dtype)
            phi = tf.cos(tf.matmul(inputs, samples) + bias) # Check this broadcasting
        else:
            phi = np.cos(np.matmul(inputs, samples) + bias)
        scaled_phi = sqrt(2.0 / self.n_rff) * phi
        return scaled_phi
    # If matern, it considers a seperable kernel on the dimensions.
    def check(self, inputs, stddev=1.0, nu=1.5):
        size = len(inputs)
        scaled_phi = self.map(inputs, stddev, nu, tf_version=False)
        rff_kernel = np.matmul(scaled_phi, scaled_phi.T)
        if self.kernel == 'gaussian':
            kernel = rbf_kernel(inputs, inputs, gamma=1.0/(2.0*stddev**2))
        elif self.kernel == 'matern':
            kernel = np.ones(shape=(size, size))
            kernel_class = Matern(length_scale=stddev, nu=nu)
            for i in range(self.in_dim):
                #print(i)
                #print(kernel_class(np.expand_dims(inputs[:,i],1)))
                kernel = np.multiply(kernel, kernel_class(np.expand_dims(inputs[:,i],1)))
        #print('Approx Kernel:', rff_kernel, np.sum(rff_kernel))
        #print('True Kernel:', kernel, np.sum(kernel))
        return rff_kernel, kernel




