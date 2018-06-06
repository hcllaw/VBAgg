from __future__ import division

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state

from kerpy.GaussianKernel import GaussianKernel
from kerpy.GaussianBagKernel import GaussianBagKernel

class Kernel_computations():
    def __init__(self, bag_kernel, indiv_kernel, ker_precompute, bag_bw=None, indiv_bw=None):
        self.bag_kernel = bag_kernel
        self.indiv_kernel = indiv_kernel
        self.ker_precompute
        if bag_bw is not None:
            self.bag_bw = bag_bw
        if indiv_bw is not None:
            self.indiv_bw = indiv_bw

    def pixel_kernel_batch(self, batch):
        raise NotImplementedError()
        if not isinstance(batch, Mal_features):
            raise TypeError('batch is not a Mal_features object')

        root_directory = 'none'
        location_path = os.path.join(root_directory, 'pixel_dist', batch.dataset_name)
        if self.ker_precompute and self.pix_kernel == 'rbf':
            bounds = np.r_[0, np.cumsum(batch.n_pts)] # join a number with np.array()
            indices = np.vstack([
                [row, col] 
                for (start, end) in zip(bounds[:-1], bounds[1:]) #[(0, 3), (3, 5), (5, 9)] for batch.npts 3,5,9
                for row in range(start, end)
                for col in range(start, end)
                ])
            values = []   
            for i, bag in enumerate(batch):
                file_name = 'euclid_dist_{}'.format(bag.indice)
                save_path = os.path.join(location_path, file_name)
                pix_kernel = np.load(save_path)
                values = values + pix_kernel.reshape(bag.n_pts*bag.npts).tolist()

            sparse_batch_kernel = SparseInfo(indices=indices, 
                                             values = values,
                                             dense_shape = [batch.total_points, batch.total_points],
                                             )
        else:
            raise ValueError('Only RBF kernel currently.')
        return sparse_batch_kernel
        '''
            def bag_kernel_batch(self, batch, num_freq=250, seed=23):
                # TODO: Implement for other kernels etc and throw exceptions
                data_gauss_kernel = GaussianKernel(sigma=pix_bw)
                gauss_kernel = LinearBagKernel(data_gauss_kernel)
                # TODO:check this is correct, works for list
                np.random.seed(seed)
                gauss_kernel.rff_generate(mdata= num_freq, dim= batch.ndim)
                batch_means = gauss_kernel.rff_expand(batch)
                batch_pair_dists = squareform(pdist(batch_means, 'sqeuclidean'))
                return batch_pair_dists
        '''
    @staticmethod
    def get_median_sqdist(feats, kernel='rbf', dims_last=None, bags=False, n_sub=5000, seed=23):
        np.random.seed(seed)
        if bags:
            all_Xs = np.concatenate(feats)
        else:
            all_Xs = feats
        N = all_Xs.shape[0]
        sub = all_Xs[np.random.choice(N, min(n_sub, N), replace=False)]
        if kernel == 'ard':
            dim = np.shape(sub)[1]
            D2 = euclidean_distances(sub, squared=True)
            med_distance_sq = np.median(D2[np.triu_indices_from(D2, k=1)], overwrite_input=True)
            return np.repeat(med_distance_sq, dim) # median square distance 
        elif kernel == 'rbf':
            D2 = euclidean_distances(sub, squared=True)
            return np.median(D2[np.triu_indices_from(D2, k=1)], overwrite_input=True)
        elif kernel == 'additive':
            D2_1 = euclidean_distances(sub[:,:-dims_last], squared=True)
            dim1 = np.shape(sub[:,:-dims_last])[1]
            md_sq_1 = np.median(D2_1[np.triu_indices_from(D2_1, k=1)], overwrite_input=True)
            bw_1 = np.repeat(md_sq_1, dim1)
            D2_2 = euclidean_distances(sub[:,-dims_last:], squared=True)
            md_sq_2 = np.median(D2_2[np.triu_indices_from(D2_2, k=1)], overwrite_input=True)
            bw_2 = np.repeat(md_sq_2, dims_last)
            return np.concatenate((bw_1, bw_2)) # For ARD + Matern 
        else:
            raise ValueError('{} not currently implemented'.format(kernel))
        

    @staticmethod
    def get_bag_median_sqdist(feats, indiv_bw, num_freq=250, seed=23):
        data_gauss_kernel = GaussianKernel(sigma=indiv_bw)
        gauss_kernel = LinearBagKernel(data_gauss_kernel)
        # TODO:check this is correct, works for list
        np.random.seed(seed)
        gauss_kernel.rff_generate(mdata= num_freq, dim= feats.ndim)
        means = gauss_kernel.rff_expand(batch)
        bag_median_sqdist = self.get_median_sqdist(means)
        return bag_median_sqdist