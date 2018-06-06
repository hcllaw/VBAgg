from __future__ import division
from functools import partial
import random

import numpy as np
from sklearn.datasets.samples_generator import make_s_curve, make_swiss_roll
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from scipy import stats


from agg_learn.utils import scaler_transform, increase_dim, standardise, swiss_norm, elevators_norm
from agg_learn.mal_features import Mal_features
from agg_learn.kernel_computations import Kernel_computations
from agg_learn.toy_low_dim import sphere, rotate_orth

# Toy Generation Classes 
class Toy:
    def __init__(self, dim=10, size_type='uniform', 
                 bag_sizes=[50,100], noise=0.0, preprocess='standardise'):
        self.dim = dim
        self.preprocess = preprocess
        self.size_type = size_type
        self.bag_sizes = bag_sizes
        self.noise = noise
    # Generate Bag sizes 
    def bag_size_gen(self, num_bags, random_state, max_pts=None):
        if self.size_type == 'uniform':
            lo, hi = self.bag_sizes
            sizes = random_state.randint(low=lo, high=hi + 1, size=num_bags)
        elif self.size_type == 'neg-binom':
            # Do a negative binomial + 1 (in Wikipedia's notation),
            # so that sizes are a distribution on the positive integers.
            # mean = p r / (1 - p) + 1; var = (mean - 1) / (1 - p)
            mean, std = self.bag_sizes
            p = 1 - (mean - 1) / (std * std)
            assert 0 < p < 1
            r = (mean - 1) * (1 - p) / p
            assert r > 0
            # scipy swaps p and 1-p
            sizes = []
            if max_pts is not None:
                while max_pts > 0:
                    size_bag = stats.nbinom(r, 1 - p).rvs(size=1, random_state=random_state)[0] + 1
                    max_pts = max_pts - size_bag
                    if max_pts >= 0:
                        sizes.append(size_bag)
                    else:
                        sizes.append(max_pts + size_bag)
            else:
                sizes = stats.nbinom(r, 1 - p).rvs(size=num_bags, random_state=random_state) + 1
        else:
            raise ValueError("unknown size_type {}".format(self.size_type))
        return sizes

    def preprocessing(self, bags, scaler, train, bw, kernel, seed=23, bag_var=False):
        if train:
            if self.preprocess == 'standardise':
                scaler = StandardScaler()
            elif args.preprocess == 'normalize':
                scaler = MinMaxScaler()
        if bag_var:
            if train:
                bags = scaler.fit_transform(bags)
                if bw is None:
                    bw = np.sqrt(Kernel_computations.get_median_sqdist(bags, bags=False, seed=seed) / 2)
            else:
                bags = scaler.transform(bags)
        else:
            scaler, bags = scaler_transform(scaler, bags, train=train)
            if train and bw is None:
                bw = np.sqrt(Kernel_computations.get_median_sqdist(bags, kernel=kernel, bags=True, seed=seed) / 2)
        return bags, scaler, bw

    def toy_s_gen(self, num_bags, scaler=None, indiv=None, bw=None, 
                  data_noise=0.2, train=False, y_type='normal', 
                  kernel='rbf', sigma=1.0, seed=23):
        rs = check_random_state(seed)
        if y_type == 'normal':
            scale = 1.0
            y_gen = partial(rs.normal, scale=sigma)
        elif y_type == 'poisson':
            scale = 0.5 
            y_gen = lambda rate: rs.poisson(rate)
        else:
            raise TypeError('y_gen type {} not understood'.format(y_type))
        sizes = self.bag_size_gen(num_bags, rs)
        total_pts = np.sum(sizes)
        X, label = make_s_curve(total_pts, noise=data_noise, random_state=rs)
        label = label + 6.0 # To ensure everything is positive
        label = scale * label 
        sort_index = X[:,2].argsort() # Last dimension is vertical axis
        X = X[sort_index[::-1]]
        label = label[sort_index[::-1]]
        X_dim = increase_dim(X, self.dim)
        data = rotate_orth(X_dim, seed=23) # Rotate into dim-dimensional object.
        indexes = [0] + np.cumsum(sizes).tolist()
        bags = []
        indiv_labels = []
        indiv_true_labels = []
        bag_true_labels = []
        bag_labels = np.zeros(num_bags)
        for i in range(num_bags):
            lower = indexes[i]
            upper = indexes[i+1]
            indiv_label_bag = [ y_gen(phi) for phi in label[lower:upper] ]
            bag_labels[i] = np.sum(indiv_label_bag)
            bags.append(data[lower:upper])
            indiv_labels.append(indiv_label_bag)
            indiv_true_labels.append(label[lower:upper])
            bag_true_labels.append(np.sum(label[lower:upper]))
        bags, scaler, bw = self.preprocessing(bags, scaler, train, bw, kernel, seed=seed)
        bags = [np.column_stack((bags[index], np.ones(len(bags[index])), indiv_true_labels[index], indiv_labels[index])) 
                                                           for index in range(num_bags)] 
        return Mal_features(bags, pop=True, indiv=True, true_indiv=True, y=bag_labels, true_y=bag_true_labels, bag_pop=sizes), scaler, bw 

    def toy_swiss_gen(self, num_bags, scaler=None, indiv=None, bw=None, 
                      data_noise=1.0, train=False, sigma=1.0, y_type='normal', 
                      kernel='rbf', seed=23):
        rs = check_random_state(seed)
        if y_type == 'normal':
            scale = 1.0
            y_gen = partial(rs.normal, scale=sigma)
        elif y_type == 'poisson':
            scale = 0.5
            y_gen = lambda rate: rs.poisson(np.abs(rate))
        else:
            raise TypeError('y_gen type {} not understood'.format(y_type))
        sizes = self.bag_size_gen(num_bags, rs)
        #print('sizes:', sizes)
        #print(sizes)
        total_pts = np.sum(sizes)
        X, label = make_swiss_roll(total_pts, noise=data_noise, random_state=rs)
        label = scale * label
        sort_index = X[:,2].argsort() # Last dimension is vertical axis
        X = X[sort_index[::-1]]
        label = label[sort_index[::-1]]
        if y_type == 'normal':
            label = swiss_norm(label)
        X_dim = increase_dim(X, self.dim)
        data = rotate_orth(X_dim, seed=23) # Rotate into dim-dimensional object.
        indexes = [0] + np.cumsum(sizes).tolist()
        bags = []
        indiv_labels = []
        indiv_true_labels = []
        bag_true_labels = []
        bag_labels = np.zeros(num_bags)
        for i in range(num_bags):
            lower = indexes[i]
            upper = indexes[i+1]
            indiv_label_bag = [ y_gen(phi) for phi in label[lower:upper] ]
            bag_labels[i] = np.sum(indiv_label_bag)
            bags.append(data[lower:upper])
            indiv_labels.append(indiv_label_bag)
            indiv_true_labels.append(label[lower:upper])
            bag_true_labels.append(np.sum(label[lower:upper]))
        bags, scaler, bw = self.preprocessing(bags, scaler, train, bw, kernel, seed=seed)
        bags = [np.column_stack((bags[index], np.ones(len(bags[index])), indiv_true_labels[index], indiv_labels[index])) 
                                                           for index in range(num_bags)] 
        return Mal_features(bags, pop=True, indiv=True, true_indiv=True, y=bag_labels, true_y=bag_true_labels, bag_pop=sizes), scaler, bw 
    
    def toy_real_gen(self, dataset, scaler=None, indiv=None, bw=None, 
                      train=False, sigma=1.0, y_type='normal', 
                      kernel='rbf', seed=23):
        rs = check_random_state(seed)
        #if y_type == 'normal':
        #    scale = 1.0
        #    y_gen = partial(rs.normal, scale=sigma)
        #elif y_type == 'poisson':
        #    scale = 2.0
        #    y_gen = lambda rate: rs.poisson(np.abs(rate))
        #else:
        #    raise TypeError('y_gen type {} not understood'.format(y_type))
        scale = 0.1
        total_pts = len(dataset)
        sizes = self.bag_size_gen(None, rs, max_pts=total_pts)
        num_bags = len(sizes)
        print('num_bags:',num_bags, 'total_pts:', total_pts)
        data = dataset[:,:18]
        #print(data[:20, -5:].tolist())
        data = np.delete(data, [14, 16], axis=1) # remove columns with nothing...
        #print(data[:20, -5:].tolist())
        label = scale * dataset[:,-1]
        sort_index = data[:,7].argsort() # diffRollRate axis
        data = data[sort_index[::-1]]
        label = label[sort_index[::-1]]
        label = elevators_norm(label)
        print(label)
        indexes = [0] + np.cumsum(sizes).tolist()
        bags = []
        indiv_labels = []
        #indiv_true_labels = []
        #bag_true_labels = []
        bag_labels = np.zeros(num_bags)
        for i in range(num_bags):
            lower = indexes[i]
            upper = indexes[i+1]
            indiv_label_bag = label[lower:upper]
            bag_labels[i] = np.sum(indiv_label_bag)
            bags.append(data[lower:upper])
            indiv_labels.append(indiv_label_bag)
        #    indiv_true_labels.append(label[lower:upper])
        #    bag_true_labels.append(np.sum(label[lower:upper]))
        bags, scaler, bw = self.preprocessing(bags, scaler, train, bw, kernel, seed=seed)
        var_init = np.var(np.divide(bag_labels, sizes))
        print('var_init', var_init)
        bags = [np.column_stack((bags[index], np.ones(len(bags[index])), indiv_labels[index])) 
                                                           for index in range(num_bags)] 
        return Mal_features(bags, pop=True, indiv=True, true_indiv=False, y=bag_labels, bag_pop=sizes), scaler, bw, var_init
    
    def toy_swiss_bag_gen(self, num_bags, scaler=None, scaler_bag=None, 
                          indiv=None, bw=None, bw_bag=None,
                          data_noise=1.0, train=False, sigma=1.0, y_type='normal', 
                          kernel='rbf', seed=23):
        rs = check_random_state(seed)
        if y_type == 'normal':
            scale = 1.0
            y_gen = partial(rs.normal, scale=sigma)
        elif y_type == 'poisson':
            scale = 0.5
            y_gen = lambda rate: rs.poisson(np.abs(rate))
        else:
            raise TypeError('y_gen type {} not understood'.format(y_type))
        sizes = self.bag_size_gen(num_bags, rs)
        #print('sizes:', sizes)
        total_pts = np.sum(sizes)
        X, label = make_swiss_roll(total_pts, noise=data_noise, random_state=rs)
        label = scale * label
        sort_index = X[:,2].argsort()
        X = X[sort_index[::-1]]
        label = label[sort_index[::-1]]
        X_dim = increase_dim(X[:,:2], self.dim)
        data = rotate_orth(X_dim, seed=23) # Rotate into dim-dimensional object.
        indexes = [0] + np.cumsum(sizes).tolist()
        # Bag Variable on Manifold generate
        indiv_var = standardise(X[:,2])
        bag_var = []
        bags = []
        indiv_labels = []
        indiv_true_labels = []
        bag_true_labels = []
        bag_labels = np.zeros(num_bags)
        for i in range(num_bags):
            lower = indexes[i]
            upper = indexes[i+1]
            #print(indiv_var[lower:upper])
            indiv_label_bag = [ y_gen(phi) for phi in label[lower:upper] ]
            bag_value = self.s_manifold(indiv_var[lower:upper], random_state=rs)
            bag_value_rep = np.tile(bag_value, (upper-lower,1))
            bag_var.append(bag_value)
            bag_labels[i] = np.sum(indiv_label_bag)
            bags.append(np.hstack((data[lower:upper], bag_value_rep)))
            indiv_labels.append(indiv_label_bag)
            indiv_true_labels.append(label[lower:upper])
            bag_true_labels.append(np.sum(label[lower:upper]))
        bag_var = np.vstack(bag_var)
        bags, scaler, bw = self.preprocessing(bags, scaler, train, bw, kernel, seed=seed)
        bag_var, scaler_bag, bw_bag = self.preprocessing(bag_var, scaler_bag, train, bw_bag, 
                                                         kernel, seed=seed, bag_var=True)
        bags = [np.column_stack((bags[index], np.ones(len(bags[index])), indiv_true_labels[index], indiv_labels[index])) 
                                                           for index in range(num_bags)] 
        return Mal_features(bags, pop=True, indiv=True, y=bag_labels, true_indiv=True, true_y=bag_true_labels, bag_var=bag_var, bag_pop=sizes), scaler, bw, scaler_bag, bw_bag
    
    def s_manifold(self, values, dim=5, n_points=10000, noise=0.0, random_state=None):
        value = np.mean(values)
        X, y = make_s_curve(n_points, noise=noise, random_state=random_state)
        s_curve_y = standardise(y)
        X = np.expand_dims(X[np.argmin(np.abs(s_curve_y - value)),:], 0)
        #X_dim = increase_dim(X, dim)
        #X = rotate_orth(X_dim, seed=23)
        return X

    def toy_sphere_gen(self, num_bags, scaler=None, indiv=None, bw=None, 
                       data_scale=2.0, latent_dim=3, train=False, y_type='normal',
                       sigma=1.0, seed=23):
        raise NotImplementedError()
        assert latent_dim >= 3
        rs = check_random_state(seed)
        if y_type == 'normal':
            y_gen = partial(rs.normal, scale=sigma)
        elif y_type == 'poisson':
            y_gen = rs.poisson
        else:
            raise TypeError('y_gen type {} not understood'.format(y_type))
        sizes = self.bag_size_gen(num_bags, seed=seed)
        total_pts = np.sum(sizes)
        norm_data, norm_sphere = sphere(total_pts, data_scale=1.0, latent_dim=latent_dim, 
                        full_dim=self.dim, seed=seed)
        data = rotate_orth(norm_sphere, seed=23)
        print(norm_data[:10,:])
        print(data[:10,:])
        mean_y = 2.0 * np.exp(2.0*norm_data[:,1]) + np.abs(norm_data[:,2]) - np.exp(2.0*data[:,3])
        indexes = [0] + np.cumsum(sizes).tolist()
        #print('indexes', indexes)
        bags = []
        indiv_labels = []
        bag_labels = np.zeros(num_bags)
        for i in range(num_bags):
            lower = indexes[i]
            upper = indexes[i+1]
            indiv_label_bag = [ y_gen(np.abs(phi)) for phi in mean_y[lower:upper] ]
            #print(indiv_label_bag)
            bag_labels[i] = np.sum(indiv_label_bag)
            #print(bag_labels[i])
            bags.append(data[lower:upper])
            indiv_labels.append(mean_y[lower:upper])
            #if i == 0:
            #    raise 'Finish'
        #print(indiv_labels[:1])
        #print(mean_y[:100])
        bags, scaler, bw = self.preprocessing(bags, scaler, train, bw)
        bags = [np.column_stack((bags[index], np.ones(len(bags[index])), indiv_labels[index])) 
                                                           for index in range(num_bags)] 
        return Mal_features(bags, pop=True, indiv=True, y=bag_labels), scaler, bw 