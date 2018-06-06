# Construct low dimensional manifold
# 1. Construct data on sphere in R^d, by normalising each row to unit norm . 
# 2. Add zeros to contruct a sphere in R^d in R^p, by putting zeros.
# 3. Rotate with random othorgonal matrix, to embed it in R^p.

from __future__ import division

import numpy as np
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt

def sphere(data_size, data_scale=2.0, latent_dim=2, full_dim=20, seed=23):
    rs = check_random_state(seed)
    zeros_dim = full_dim - latent_dim
    latent_data = rs.normal(loc=0.0, scale=data_scale, size=((data_size, latent_dim)))
    norm = np.linalg.norm(latent_data, ord=2, axis=1)
    norm_data = latent_data / np.expand_dims(norm, 1)
    #plt.scatter(norm_data[:,0], norm_data[:,1])
    #plt.show()
    zeros_data = np.zeros((data_size, zeros_dim))
    norm_sphere = np.hstack((norm_data, zeros_data))
    return norm_data, norm_sphere

def rotate_orth(data, return_orth=False, transpose=False, seed=23):
    rs = check_random_state(seed)
    dim = data.shape[1]
    random_matrix = rs.uniform(0, 2, (dim, dim))
    orth_matrix, R = np.linalg.qr(random_matrix)
    if transpose:
        orth_matrix = orth_matrix.T
    trans_data = np.dot(data, orth_matrix)
    if return_orth:
        return trans_data, orth_matrix
    else:
        return trans_data

if __name__ == '__main__': # Check
    norm_sphere = sphere(100)
    data = rotate_orth(norm_sphere)