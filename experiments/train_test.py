#!/usr/bin/env python
from __future__ import division, print_function
import argparse
from functools import partial
import os
import sys
import time

import numpy as np
import tensorflow as tf
import multiprocessing as mp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans
from sklearn.cluster.k_means_ import _init_centroids

from vb_agg_learn.mal_features import Mal_features
from vb_agg_learn.networks import (nn_net, indiv_net, baseline_net, vb_net, 
                                gp_map_net, vb_norm_net, 
                                vb_net_square)
from vb_agg_learn.train import eval_network, train_network, baselines
from vb_agg_learn.utils import (tf_session, scaler_transform, true_dimension,
                             load_malaria, check_positive, extract, 
                             load_split_elevators, partition_maker)
from vb_agg_learn.kernel_computations import Kernel_computations

def get_adder(g):
    def f(*args, **kwargs):
        kwargs.setdefault('help', "Default %(default)s.")
        return g.add_argument(*args, **kwargs)
    return f

def _add_args(subparser):
    network = subparser.add_argument_group("Network parameters")
    n = get_adder(network)
    n('--net-type', choices=['nn', 'indiv', 'baseline',
                             'vb', 'gp_map'], default='baseline')
    n('--link', choices=['exp', 'square'], default='exp')
    n('--structure', type=int, default=20, help='No. of landmarks (if landmark-choice is bag) \
                                                 or number of hidden neurons')
    n('--landmark-choice', choices=['bag', 'all'], default='all', help='bag is for landmark per bag') 
    n('--landmarks-select', choices=['kmeans', 'kmeans++'], default='kmeans++')
    n('--n-landmark-bag', type=int, default=1, help='No. of landmarks when landmark-choice is bag')
    train = subparser.add_argument_group("Training parameters")
    t = get_adder(train)
    t('--max-epochs', type=int, default=100)
    int_inf = lambda x: np.inf if x.lower() in {'inf', 'none'} else int(x)
    t('--batch-pts', type=int_inf, default='inf') # Number of pts per patch
    t('--batch-bags', type=int_inf, default=15) 
    t('--eval-batch-bags', type=int_inf, default=15) # Evaluation usage, care for large data size.
    t('--eval-batch-pts', type=int_inf, default='inf')
    t('--learning-rate', '--lr', type=float, default=0.01) 
    t('--dtype-double', action='store_true', default=True) # Use if inverse/cholesky issues.
    t('--dtype-single', action='store_false', dest='dtype_double')
    t('--optimizer', choices=['adam', 'sgd'], default='adam')
    t('--estop-criterion', choices=['mse', 'nll'], default='nll') 
    t('--tune', choices=['bag','indiv'], default='bag') # Tuning on bags or indiv, indiv provide baseline.
    t('--initialse', choices=['identity', 'kernel'], default='kernel') # For vb, useful for safe intialisation.
    t('--gradient-clip', action='store_true', default=False) # If gradients explode
    t('--vb-mean', action='store_true', default=True) # intialise from mean 
    t('--opt-seed', type=int, default=np.random.randint(2**23))
    #t('--MAP', action='store_true', default=False)
    G1 = network.add_mutually_exclusive_group()
    g1 = get_adder(G1)
    g1('--no-early-stop', action='store_true', default=False) # no early stopping may break...
    g1('--first-early-stop-epoch', type=int, help="Default: MAX_EPOCHS / 3.")

    reg = network.add_argument_group('Regularisation')
    r = get_adder(reg)
    r('--reg-indiv', type=float, default=0.0) # Laplacian Reg for individuals
    r('--reg-bag', type=float, default=0.0) # Laplacian Reg for Bags
    r('--reg-out', type=float, default=0.0) # L2 loss reg
    
    kernel = subparser.add_argument_group("Kernel parameters")
    k = get_adder(kernel)
    k('--bw-scale', type=float, default=1.0) # scaling for indiv bw
    k('--bw-bag-scale', type=float, default=1.0) # scaling for bag bw
    k('--bw-indiv', type=float, default=None,
                help='RBF indiv kernel bw, default \
                median heuristic bandwidth') 
    k('--bw-bag', type=float, default=None,
                help='RBF bag kernel bw, default \
                median heuristic bandwidth \
                wrt the indiv kernel')
    k('--bag-kernel', choices=['rbf'], default='rbf') 
    k('--indiv-kernel', choices=['rbf', 'ard', 'additive'], default='rbf') # additive for malaria dataset
    k('--approx-kernel', choices=['rff'], default='rff') #remove landmark for ambiguity
    k('--n-rff', type=int, default=500) #sqrt of data points from theory?
    
    io = subparser.add_argument_group("I/O parameters")
    i = get_adder(io)
    io.add_argument('out_dir') # TODO: Current intialisation CPUs is 1 only...
    i('--n-cpus', type=int, default=min(1, mp.cpu_count()))  
    i('--gpu', type=int, default=None) # Set 0-7 

def make_parser(rest_of_args=_add_args):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="The dataset to run on")
    # Subparser chosen by the first argument of your parser
    def add_subparser(name, **kwargs):
        subparser = subparsers.add_parser(name, **kwargs)
        subparser.set_defaults(dataset=name)
        data = subparser.add_argument_group('Data parameters')
        rest_of_args(subparser)
        return data, get_adder(data)

    def add_sim_args(g): # For Simulated/Toy Datasets 
        a = g.add_argument
        a('--preprocess', choices=['None', 'standardise', 'normalize'], default='standardise')
        a('--n-train', type=int, default=300)
        a('--n-estop', type=int, default=100)
        a('--n-val',   type=int, default=100)
        a('--n-test',  type=int, default=500)
        a('--dim', '-d', type=int, default=18)
        a('--data-seed', type=int, default=np.random.randint(2**32))
        a('--size-type', choices=['uniform', 'neg-binom'], default='neg-binom')
        a('--size-mean', type=int, default=50, help='size mean for neg-binomial')
        a('--size-std', type=int, default=25, help='size sd for neg-binomial')
        a('--min-size', type=int, default=50, help='min bag size')
        a('--max-size', type=int, default=50, help ='max bag size')

    def add_split_args(g): # For Real Datasets 
        a = g.add_argument
        a('--preprocess', choices=['None', 'standardise', 'normalize'], default='standardise')
        a('--split', choices=['stratify', 'random'], default='stratify')
        a('--test-size', type=check_positive, default=.2,
          help="Number or portion of overall data to use for testing "
               "(default %(default)s).")
        a('--val-size', type=check_positive, default=.1,
          help="Number or portion of overall data to use for validation, "
               "(default %(default)s).")
        a('--estop-size', type=check_positive, default=.1,
          help="Number or portion of overall data to use for estop "
               "(default %(default)s).")
        a('--train-size', type=check_positive, default=None, # Will be rest 
          help="Number or portion of overall data to use for training "
               "(default complement of other sizes).")
        a('--split-seed', type=int, default=np.random.randint(2**32),
          help="Seed for the split process (default: random).")

    # http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html
    swiss, d = add_subparser('swiss') # Sklearn Swiss Roll Dataset
    d('--data-noise', type=float, default=0.0) # Add noise onto data 
    d('--y-gen-type', choices=['normal', 'poisson'], default='poisson') 
    d('--sigma', type=float, default=0.1) # For normal y-gen-type
    add_sim_args(swiss)

    swiss_bag, d = add_subparser('swiss_bag') # Swiss Roll Dataset (Modifed with bag variable)
    d('--data-noise', type=float, default=0.0)
    d('--y-gen-type', choices=['normal', 'poisson'], default='poisson')
    d('--sigma', type=float, default=1.0) #For normal y-gen-type
    add_sim_args(swiss_bag)

    # http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_s_curve.html
    s_shape, d = add_subparser('s_shape') # S Shape dataset 
    d('--data-noise', type=float, default=0.0)
    d('--y-gen-type', choices=['normal', 'poisson'], default='poisson')
    d('--sigma', type=float, default=1.0)
    add_sim_args(s_shape)

    # Malaria Dataset
    malaria, d = add_subparser('malaria')
    d('--dataset-name', '-d', default='malaria_13')
    d('--y-gen-type', choices=['poisson'], default='poisson')
    add_split_args(malaria)

    # Elevators Dataset 
    elevators, d = add_subparser('elevators')
    d('--y-gen-type', choices=['normal', 'poisson'], default='normal')
    d('--sigma', type=float, default=1.0)
    add_sim_args(elevators)
    return parser

def check_output_dir(dirname, parser, make_checkpoints=False):
    checkpoints = os.path.join(dirname, 'checkpoints')
    if os.path.exists(dirname):
        files = set(os.listdir(dirname)) - {'output'}
        if 'checkpoints' in files:
            if not os.listdir(checkpoints):
                files.discard('checkpoints')
        if files:
            parser.error(("Output directory {} exists, and I'm real scared of "
                          "overwriting data. Change the name or delete it.")
                         .format(dirname))
    else:
        os.makedirs(dirname)
    if make_checkpoints and not os.path.isdir(checkpoints):
        os.makedirs(checkpoints)

def parse_args(rest_of_args=_add_args):
    parser = make_parser(rest_of_args)
    args = parser.parse_args()
    check_output_dir(args.out_dir, parser, make_checkpoints=True)
    return args

# Split Real Data (Includes population + indexes)
def _split_feats(args, feats_x, indiv_pop, feats_y, feats_index, bag_pop, missing=False):
    if args.estop_size !=0 and args.no_early_stop:
        raise ValueError('Cannot specify no early stop and estop size as {}'.format(args.estop_size))
    total_size = args.test_size + args.val_size + args.estop_size 
    if total_size >= 1.0:
        raise ValueError('Total proportions of val, estop, test >= 1.0')
    rs = check_random_state(args.split_seed)
    size = len(feats_x)
    if args.split == 'random':
        permute = rs.permutation(size)
        test_s = int(args.test_size * size)
        val_s = int(args.val_size * size)
        if args.no_early_stop:
            estop_s = 0
        else:
            estop_s = int(args.estop_size * size)
        train_s = size - test_s - val_s - estop_s
        train_ind = permute[:train_s]
        val_ind = permute[train_s: train_s + val_s]
        estop_ind = permute[train_s + val_s: train_s + val_s + estop_s]
        test_ind = permute[train_s + val_s + estop_s:]
    elif args.split == 'stratify':
        train_ind, val_ind, estop_ind, test_ind = partition_maker(feats_y, bag_pop, args.estop_size, 
                                                                  args.val_size, args.test_size, 
                                                                  rs)
        # For simplicity, we order labels then we split into 10 parts, adding 
        # each one to a bag.
    else:
        raise ValueError('{} not recognized'.format(args.split))
    if args.preprocess == 'standardise':
        scaler = StandardScaler()
    elif args.preprocess == 'normalize':
        scaler = MinMaxScaler()
    else:
        scaler = None
    scaler, train_x = scaler_transform(scaler, [feats_x[i] for i in train_ind], train=True, coords=True)

    st = partial(scaler_transform, scaler, coords=True)
    _, val_x = st([feats_x[i] for i in val_ind])
    _, estop_x = st([feats_x[i] for i in estop_ind])
    _, test_x = st([feats_x[i] for i in test_ind])

    if args.bw_indiv is None:
        if args.indiv_kernel == 'rbf':
            args.bw_indiv = np.sqrt(Kernel_computations.get_median_sqdist(train_x, kernel='rbf', bags=True, seed=args.split_seed) / 2)
        elif args.indiv_kernel == 'ard':
            args.bw_indiv = np.sqrt(Kernel_computations.get_median_sqdist(train_x, kernel='ard', bags=True, seed=args.split_seed) / 2)
        elif args.indiv_kernel == 'additive':
            args.bw_indiv = np.sqrt(Kernel_computations.get_median_sqdist(train_x, kernel='additive', dims_last=2, bags=True, seed=args.split_seed) / 2)
    
    l = lambda x, pop, ind: [np.column_stack((x[index], pop[i])) for index, i in enumerate(ind)]

    bag_pop = np.array(bag_pop)
    # Make a class of data, labels, population... 
    train = Mal_features( l(train_x, indiv_pop, train_ind), pop=True, y=feats_y[train_ind],
                            bag_pop= bag_pop[train_ind], ind=train_ind)
    train_index = [feats_index[i] for i in train_ind]
    val = Mal_features( l(val_x, indiv_pop, val_ind), pop=True, y=feats_y[val_ind], 
                          bag_pop= bag_pop[val_ind], ind=val_ind)
    val_index = [feats_index[i] for i in val_ind]
    if args.no_early_stop:
        estop = None
        estop_index = None
    else:
        estop = Mal_features( l(estop_x, indiv_pop, estop_ind),pop=True, y=feats_y[estop_ind], 
                                bag_pop= bag_pop[estop_ind], ind=estop_ind)
        estop_index = [feats_index[i] for i in estop_ind]
    test = Mal_features( l(test_x, indiv_pop, test_ind), pop=True, y=feats_y[test_ind],
                            bag_pop= bag_pop[test_ind], ind=test_ind)
    test_index = [feats_index[i] for i in test_ind]
    indexes = (train_index, estop_index, val_index, test_index)
    if missing:
        return train, estop, val, test, indexes, scaler
    else:
        return train, estop, val, test, indexes

def generate_data(args):
    # Toy + Bagging Elevators dataset
    if args.dataset in ['s_shape', 'swiss', 'swiss_bag', 'elevators']:
        from vb_agg_learn.data.toy import Toy
        d = dict(
            dim=args.dim,
            preprocess=args.preprocess,
        )
        if args.size_type == 'uniform':
            d['size_type'] = 'uniform'
            d['bag_sizes'] = [args.min_size, args.max_size]
        elif args.size_type == 'neg-binom':
            d['size_type'] = 'neg-binom'
            d['bag_sizes'] = [args.size_mean, args.size_std]
        else:
            raise ValueError("unknown size_type {}".format(args.size_type))

        toy = Toy(**d)
        rs = check_random_state(args.data_seed)
        args.train_seed, args.estop_val_seed, args.optval_seed, args.test_seed \
            = rs.randint(2**32, size=4) # Seeds for generation 
        if args.dataset != 'swiss_bag': 
            if args.dataset == 'swiss': 
                make = partial(toy.toy_swiss_gen, data_noise=args.data_noise, 
                               y_type=args.y_gen_type, sigma=args.sigma)
            elif args.dataset == 's_shape':
                make = partial(toy.toy_s_gen, data_noise=args.data_noise,
                               y_type=args.y_gen_type, sigma=args.sigma)
            elif args.dataset == 'elevators': # Elevators Split 
                if args.no_early_stop:
                    es_split = 0.0
                else:
                    es_split = 0.1 
                train, estop, val, test = load_split_elevators(es_plit=es_split, val_split=0.1, 
                                                               test_split=0.1, split_seed=args.data_seed)
                make = partial(toy.toy_real_gen, y_type=args.y_gen_type, sigma=args.sigma)
            else:
                raise ValueError('Unknown toy_type {}'.format(args.dataset))
            if args.dataset == 'elevators':
                train, scaler, args.bw_indiv, args.var_init = make(train, train=True, bw=args.bw_indiv, 
                                                kernel=args.indiv_kernel, seed=args.train_seed)
                estop, _, _, _ = make(estop, scaler = scaler, train=False, seed=args.estop_val_seed)
                val, _, _, _ = make(val, scaler = scaler, train=False, seed=args.optval_seed)
                test, _, _, _ = make(test, scaler = scaler, train=False, seed=args.test_seed)
            else:
                train, scaler, args.bw_indiv = make(args.n_train, train=True, bw=args.bw_indiv, 
                                                    kernel=args.indiv_kernel, seed=args.train_seed)
                estop, _, _ = make(args.n_estop, scaler = scaler, train=False, seed=args.estop_val_seed)
                val, _, _ = make(args.n_val, scaler = scaler, train=False, seed=args.optval_seed)
                test, _, _ = make(args.n_test, scaler = scaler, train=False, seed=args.test_seed)
        elif args.dataset == 'swiss_bag':
            make = partial(toy.toy_swiss_bag_gen, data_noise=args.data_noise,
                           y_type=args.y_gen_type, sigma=args.sigma)
            train, scaler, args.bw_indiv, scaler_bag, args.bw_bag = make(args.n_train, train=True, 
                                                                         bw=args.bw_indiv, bw_bag=args.bw_bag, 
                                                                         seed=args.train_seed)
            estop, _, _, _, _ = make(args.n_estop, scaler = scaler, scaler_bag=scaler_bag, 
                                     train=False, seed=args.estop_val_seed)
            val, _, _, _, _ = make(args.n_val, scaler = scaler, scaler_bag=scaler_bag, 
                                   train=False, seed=args.optval_seed)
            test, _, _, _, _ = make(args.n_test, scaler = scaler, scaler_bag=scaler_bag, 
                                    train=False, seed=args.test_seed)
        else:
            raise ValueError('Unknown toy_type {}'.format(args.dataset))
        indexes = None
    elif args.dataset == 'malaria':
        feats_x, feats_pop, feats_index, feats_y, bag_pop = load_malaria(args.dataset_name)
        train, estop, val, test, indexes = _split_feats(args, feats_x, feats_pop, 
                                                        feats_y, feats_index, bag_pop)
    else:
        raise ValueError("unknown dataset {}".format(args.dataset))
    #print(train, estop, val, test)
    return train, estop, val, test, indexes 

def pick_landmarks(args, train, dim): # Choose landmarks, we use opt_seed here.
    train.make_stacked()
    rs = check_random_state(args.opt_seed)
    data_fit = train.stacked_features[:,:dim] 
    if args.landmark_choice == 'all': # Sample landmarks from all pts
        if args.landmarks_select == 'kmeans++': 
            landmarks = _init_centroids(data_fit, args.structure, 'k-means++', random_state=rs)
        elif args.landmarks_select == 'kmeans':
            kmeans = KMeans(n_clusters=args.structure, random_state=rs)
            landmarks = kmeans.fit(data_fit).cluster_centers_
    elif args.landmark_choice == 'bag': # Sample landmarks per bag
        size = len(train)
        l_size = args.n_landmark_bag
        landmarks = np.zeros((size * l_size, dim))
        for i in range(size):
            bag = train[i]
            bag_x = bag[:,:dim]
            if l_size >= 1:
                landmarks[i*l_size: (i+1)*l_size] = _init_centroids(bag_x, args.n_landmark_bag, 
                                                                    'k-means++', random_state=rs)
            else:
                raise ValueError('n_landmark_bag: Must be positive > 0')
    return landmarks

# Build tensorflow network
def make_network(args, train, landmarks=None, device_name=None):
    # Initialise with avg_label
    if args.dataset == 'malaria':
        avg_label = np.sum(train.y)/np.sum(train.bag_pop)
    else: #Toy dataset, population always 1 here, change accordingly
        avg_label = np.sum(train.y)/train.total_points
    true_dim = true_dimension(train)
    if hasattr(train, 'bag_var'): #Bag covariates
        bag_var = True
    else:
        bag_var = False
        args.bw_bag_scale = None
    if args.net_type in ['vb', 'gp_map']:
        if args.landmark_choice == 'bag':
            args.structure = len(train) * args.n_landmark_bag
        train.make_stacked()
        data = train.stacked_features[:,:true_dim]
        if hasattr(args, 'train_seed'):
            seed = args.train_seed
        else:
            seed = args.split_seed
        bw_indiv_L = np.sqrt(Kernel_computations.get_median_sqdist(data, kernel='rbf', bags=False, seed=seed) / 2)
        # bw for laplacia  for VB ang GPMAP
    else:
        bw_indiv_L = None
    # Dictionary for configurations 
    kw = dict(
        in_dim=true_dim, n_hidden=args.structure, data_type=args.y_gen_type, link=args.link,
        reg_indiv=args.reg_indiv, reg_bag=args.reg_bag, reg_out=args.reg_out,
        bw_indiv=args.bw_indiv, bw_bag=args.bw_bag,
        bw_scale=args.bw_scale, bw_bag_scale=args.bw_bag_scale, bw_indiv_L=bw_indiv_L,
        bag_reg=bag_var, total_size=len(train),
        approx_kernel=args.approx_kernel, kernel=args.indiv_kernel,
        n_rff=args.n_rff,
        indiv_y_bol=train.indiv,
        seed=args.opt_seed,
        landmarks=landmarks,
        landmarks_size=len(landmarks) if landmarks is not None else 30, # hack 
        initialse=args.initialse,
        dtype=tf.float64 if args.dtype_double else tf.float32,
        var_init= args.var_init if hasattr(args, 'var_init') else 0.01,
        log_y = True if args.dataset == 'malaria' else False, device_name=device_name,
        avg_label = avg_label if args.vb_mean else 1.0,
    )
    if args.net_type == 'nn':
        return nn_net.build_net(**kw)
    elif args.net_type == 'indiv':
        if train.indiv:
            return indiv_net.build_net(**kw)
        else:
            raise ValueError('Need to know individuals to train this network.')
    elif args.net_type == 'baseline':
        return baseline_net.build_net(**kw)
    elif args.net_type == 'vb':
        if args.y_gen_type == 'normal':
            return vb_norm_net.build_net(**kw)
        elif args.y_gen_type == 'poisson':
            if args.link == 'exp':
                return vb_net.build_net(**kw)
            elif args.link == 'square':
                return vb_net_square.build_net(**kw)
    elif args.net_type == 'gp_map':
        return gp_map_net.build_net(**kw)
    else:
        raise ValueError('Specify normal or poisson net.')

# Training
def train_net(sess, args, net, train, estop, landmarks=None):
    optimizer = {
        'adam': tf.train.AdamOptimizer,
        'sgd': tf.train.GradientDescentOptimizer,
    }[args.optimizer]
    cur_min = train_network(sess, net, train, estop,
                  os.path.join(args.out_dir, 'checkpoints/model'),
                  batch_pts=args.batch_pts, batch_bags=args.batch_bags,
                  eval_batch_pts=args.eval_batch_pts,
                  eval_batch_bags=args.eval_batch_bags,
                  estop_criterion = args.estop_criterion,
                  max_epochs=args.max_epochs, net_type=args.net_type,
                  early_stop= not args.no_early_stop,
                  first_early_stop_epoch=args.first_early_stop_epoch,
                  optimizer=optimizer, tune=args.tune, landmarks=landmarks,
                  lr=args.learning_rate, seed=args.opt_seed, 
                  gradient_clip=args.gradient_clip)
    return cur_min

# Evaluation 
def eval_net(args, net, fold, landmarks=None, criterion=None):
    return eval_network(net, fold,
                        batch_pts=args.eval_batch_pts,
                        batch_bags=args.eval_batch_bags,
                        net_type=args.net_type, landmarks=landmarks,
                        criterion=criterion, gen_type=args.y_gen_type,
                        log_y=True if args.dataset == 'malaria' else False,
                        link=args.link)
# Main function
def main():
    args = parse_args()
    # GPU
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu)
        device_name = '/device:GPU:{}'.format(args.gpu)
        print('Using {}'.format(device_name))
    else: # Single CPU usage 
        device_name = '/cpu:0' 
    print("Loading data...")
    # Data Generation
    train, estop, val, test, indexes = generate_data(args)
    # Timer 
    start = time.time()
    # Landmarks for vb, gp or landmarks for laplacian in NN (using Nystrom)
    if args.net_type in ['vb', 'gp_map'] or (args.net_type == 'nn' and args.indiv_kernel=='additive'):
        true_dim = true_dimension(train)
        # population for individual/labels attached 
        landmarks = pick_landmarks(args, train, true_dim)
    else:
        landmarks = None
    print("Constructing network...")
    net = make_network(args, train, landmarks)

    d = {'args': args}
    np.set_printoptions(precision=4, suppress=True)
    with tf_session(n_cpus=args.n_cpus, gpu=args.gpu) as sess:
        cur_min = train_net(sess, args, net, train, estop, landmarks)
        elapsed = (time.time() - start) # Training time
        criterion_list = ['bag']
        if train.indiv:
            criterion_list = criterion_list + ['indiv'] # Include individual results for Toy
        for name, ds in [('Train', train),('Estop', estop), ('Val', val), ('Test', test)]:
            print('-'*30)
            print(name)
            for type_data in criterion_list:
                print('{} level'.format(type_data))
                # TODO: CLEAN 
                if type_data == 'indiv': 
                    log_y = False 
                    ds.make_stacked()
                    if train.true_indiv: # Note ordering placement, true_y, observe_y
                        true_y = ds.stacked_features[:,-2]
                    else: # observed y at the end
                        true_y = ds.stacked_features[:,-1]
                elif type_data == 'bag':
                    if hasattr(ds, 'true_y'): # True Bag y?
                        true_y = ds.true_y 
                        use_true=True
                        log_y=False 
                    else:
                        true_y = ds.y 
                        use_true=False
                        if args.dataset == 'malaria': # avoid log 0 issue.
                            true_y[true_y <= 0] = 1.0
                            true_y = np.log(true_y) # log scale comparisons. 
                            log_y = True
                        else:
                            log_y = False
                y = eval_net(args, net, ds, landmarks=landmarks, criterion=type_data) # Prediction y
                nll = eval_net(args, net, ds, landmarks=landmarks, criterion='{}_nll'.format(type_data)) #NLL
                if args.y_gen_type == 'poisson': # Poisson Baselines for NLL, normal baseline (do not have sigma MLE)
                    base_nll = baselines(train, ds, criterion=type_data, true_y=use_true,
                                         net=args.net_type, choice='nll', log_y=log_y,
                                         dtype=tf.float64 if args.dtype_double else tf.float32)
                    d[name + '_{}_nll_baseline'.format(type_data)] = base_nll
                    print('Baseline {:s} NLL: {}'.format(type_data, base_nll))
                base_mse = baselines(train, ds, criterion=type_data, true_y=use_true,
                                     net=args.net_type, choice='mse', log_y=log_y,
                                     dtype=tf.float64 if args.dtype_double else tf.float32)
                #print('pred_y', y[:50])
                #print(np.mean(np.square(y - true_y)))
                d[name + '_{}_mse_baseline'.format(type_data)] = base_mse
                if args.net_type == 'vb' and type_data == 'indiv' and train.true_indiv:
                    calibrate = eval_net(args, net, ds, landmarks=landmarks, criterion='calibration')
                    d[name + '_calibration'] = calibrate
                    print('Calibration: {}'.format(calibrate)) # model calibration
                d[name + '_{}_nll'.format(type_data)] = nll
                d[name + '_{}_mse'.format(type_data)] = mse = mean_squared_error(true_y, y)
                # note this is R2 with within bag avg for indiv, for bag it accounts for population.
                d[name + '_{}_r2'.format(type_data)] = r2 = 1 - mse / base_mse 
                print('Baseline {:s} MSE: {}'.format(type_data, base_mse))
                #print('Baseline {:s} Avg NLL: {:.6f}'.format(type_data, base_nll))
                print('{} MSE: {}'.format(type_data, mse))
                print('{} Avg NLL: {}'.format(type_data, nll))
                print('{} R2 : {}\n'.format(type_data, r2))
        d['train_loss'] = cur_min
        d['time_train'] = elapsed
        #print(d)
    np.savez(os.path.join(args.out_dir, 'results.npz'), **d)

if __name__ == '__main__':
    main()
