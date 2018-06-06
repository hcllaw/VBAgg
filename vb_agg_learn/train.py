from __future__ import division, print_function
from functools import partial

import numpy as np
import tensorflow as tf
from sklearn.externals.six.moves import xrange
from sklearn.utils import check_random_state
import scipy.stats as ss

from .utils import loop_batches
from networks.vb_utility import log_normal_modes, log_norm_quantile

def train_network(sess, net, train_f, estop_f, checkpoint_path,
                  batch_pts, batch_bags=np.inf, estop_criterion ='mse',
                  eval_batch_pts=None, eval_batch_bags=None, lr=0.01, landmarks=None,
                  early_stop=True, first_early_stop_epoch=np.inf, max_epochs=1000, net_type=None,
                  optimizer=tf.train.AdamOptimizer, tune='bag', display_every=1, gradient_clip=False, 
                  seed=23):
    
    def feed_network(fold, landmarks=None, optimize=False):
        if optimize:
            looper = train_looper
        else:
            looper = eval_looper
        fold_loss = 0
        fold_bag_nll = 0
        fold_indiv_nll = 0
        fold_bag_se = 0
        fold_indiv_se = 0
        for batch_i, batch in enumerate(looper(fold)):
            if optimize:
                 _, loss, bag_nll, indiv_nll, bag_se, indiv_se = sess.run(
                        [optimize_step, net.loss, net.bag_nll, 
                         net.indiv_nll, net.bag_se, net.indiv_se],
                         feed_dict=net.feed_dict(batch, batch.y, landmarks))
            else:
                loss, bag_nll, indiv_nll, bag_se, indiv_se = sess.run(
                      [net.loss, net.bag_nll, net.indiv_nll, net.bag_se, net.indiv_se],
                       feed_dict=net.feed_dict(batch, batch.y, landmarks))
            if train_f.indiv:
                fold_indiv_se += indiv_se
                fold_indiv_nll += indiv_nll 
            fold_bag_nll += bag_nll
            fold_bag_se += bag_se
            fold_loss += loss
        fold_loss = fold_loss / (batch_i + 1)
        fold_bag_nll = fold_bag_nll / len(fold)
        fold_indiv_nll = fold_indiv_nll / fold.total_points
        fold_bag_se = fold_bag_se / len(fold)
        fold_indiv_se =  fold_indiv_se / fold.total_points
        return fold_loss, fold_bag_nll, fold_indiv_nll, fold_bag_se, fold_indiv_se

    if early_stop and first_early_stop_epoch is None:
        first_early_stop_epoch = max_epochs // 3
    if eval_batch_pts is None:
        eval_batch_pts = batch_pts
    if eval_batch_bags is None:
        eval_batch_bags = batch_bags

    rs = check_random_state(seed)
    train_looper = partial(
        loop_batches, max_pts=batch_pts, max_bags=batch_bags,
        stack=True, shuffle=True, rs=rs)
    eval_looper = partial(
        loop_batches, max_pts=eval_batch_pts, max_bags=eval_batch_bags,
        stack=True, shuffle=False)

    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #with tf.control_dependencies(update_ops):
    # Make sure we do update_ops, e.g. for batch norm, before stepping
    if gradient_clip:
        optimizer = optimizer(lr)
        gradients, variables = zip(*optimizer.compute_gradients(net.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0) #standard amount
        optimize_step = optimizer.apply_gradients(zip(gradients, variables))
    else:
        optimize_step = optimizer(lr).minimize(net.loss)

    cur_min = np.inf  # used for early stopping
    countdown = np.inf 

    saver = tf.train.Saver(max_to_keep=1)
    sess.run(tf.global_variables_initializer())
    if early_stop:
        print("Training up to {} epochs; considering early stopping from epoch {}."
              .format(max_epochs, first_early_stop_epoch))
    else:
        estop_f = train_f # For intialisation quantity.
    # Print out initialization quality

    estop_loss, estop_bag_nll, estop_indiv_nll, estop_bag_se, estop_indiv_se = feed_network(estop_f, landmarks=landmarks)
    s = "\nBefore training (on estop, if early stop):\n"
    s += "BAG: loss = {}, nll = {}, RMSE = {}\n".format(
                                          estop_loss, estop_bag_nll, np.sqrt(estop_bag_se))
    if train_f.indiv:
        s += "INDIVIDUAL: train nll = {}, train RMSE = {}\n".format(estop_indiv_nll, 
                                                                    np.sqrt(estop_indiv_se))
        s += "-"*30
    print(s)

    # Training loop
    epoch = -np.inf
    # use xrange for memory efficient 2.7 a iterator object evaluate on spot, range creates an actual list
    for epoch in xrange(max_epochs):
        train_loss, train_bag_nll, train_indiv_nll, train_bag_se, train_indiv_se = feed_network(train_f, landmarks, optimize=True)
        if early_stop:
            if epoch >= first_early_stop_epoch or epoch % display_every == 0:
                estop_loss, estop_bag_nll, estop_indiv_nll, estop_bag_se, estop_indiv_se = feed_network(estop_f, landmarks=landmarks)
                if net_type == 'vb':
                    criterion = estop_bag_nll
                    estop_criterion = 'Train Loss'
                    # we renamed estop_criterion!
                if estop_f.indiv and tune == 'indiv':
                    if estop_criterion == 'mse':
                        criterion = estop_indiv_se
                    elif estop_criterion == 'nll':
                        criterion = estop_indiv_nll
                else: # if estop_criterion is renamed, this wont 'activate'
                    if estop_criterion == 'mse':
                        criterion = estop_bag_se
                    elif estop_criterion == 'nll':
                        criterion = estop_bag_nll
                print('estop_criterion', estop_criterion)
                print('criterion', criterion)
                # Early Stopping
                if criterion <= cur_min and epoch >= first_early_stop_epoch:
                    if net_type == 'vb': # incase we want different countdown for variational
                        countdown = 10
                    else:
                        countdown = 10
                    cur_min = criterion
                    save_path = saver.save(sess, checkpoint_path)
                    best_epoch = epoch
                else:
                    countdown -= 1

        if epoch % display_every == 0:
            s = "EPOCH {:d}/{:d} (subject to early stop potentially):\n".format(epoch, max_epochs)
            s += "BAG: train loss = {}, train nll = {}, train RMSE = {}\n".format(
                                                        train_loss, train_bag_nll, np.sqrt(train_bag_se))
            if train_f.indiv:
                s += "INDIVIDUAL: train nll = {}, train RMSE = {}\n\n".format(
                                                        train_indiv_nll, np.sqrt(train_indiv_se))
            if early_stop:
                s += "Early Stop (Criterion = {} with {} if exist):\n".format(estop_criterion, tune)
                s += "Criterion: {}".format(criterion)
                s += "BAG: estop nll = {} estop RMSE = {}\n".format(estop_bag_nll, np.sqrt(estop_bag_se))
                if estop_f.indiv:
                    s += "INDIVIDUAL: estop nll = {}, estop RMSE = {}\n".format(
                                                        estop_indiv_nll, np.sqrt(estop_indiv_se))
            s += "-"*30
            print(s)

        if epoch >= first_early_stop_epoch and countdown <= 0 and early_stop:
            break

    if epoch >= first_early_stop_epoch and early_stop:
        print(("Stopping at epoch {:d} with estop criterion {}\n"
               "Using model from epoch {:d} with estop criterion as {}").format(
                   epoch, criterion, best_epoch, cur_min))
        saver.restore(sess, save_path)
    else:
        print("Using final model.")
    print('-'*30)
    # i.e the last previous model updated....
    return cur_min
    
def baselines(train_f, test_f, criterion='indiv', net='normal', choice='nll', true_y=False, log_y=False, dtype=tf.float32):
    if criterion == 'indiv':
        # y/bag_population is base_preds (discounted for population!)
        base_preds = np.array([ y/test_f.bag_pop[index] for index, y in enumerate(test_f.y) 
                                                for i in range(test_f.bag_pop[index]) ])
        test_f.make_stacked()
        if test_f.true_indiv:
            y = test_f.stacked_features[:,-2]
        else:
            y = test_f.stacked_features[:,-1]
        #print('base_y', base_preds[:50])
        #print('true_y', y[:50])
        if test_f.pop:
            if test_f.true_indiv:
                pop = test_f.stacked_features[:,-3]
            else:
                pop = test_f.stacked_features[:,-2]
            base_preds = np.multiply(base_preds, pop)
        if choice == 'nll':
            c = partial(tf.constant, dtype=dtype)
            y = test_f.stacked_features[:,-1]
            nll = tf.reduce_sum(tf.nn.log_poisson_loss(c(y), tf.log(c(base_preds)), 
                                               compute_full_loss=True))
            base_value = nll.eval()/test_f.total_points
        elif choice == 'mse':
            from sklearn.metrics import mean_squared_error
            base_value = mean_squared_error(y, base_preds)
    elif criterion == 'bag':
        avg_all = np.sum(train_f.y)/np.sum(train_f.bag_pop)
        #print('avg_all', avg_all)
        base_preds = [ avg_all * size for size in test_f.bag_pop] # adjusted for size.
        #print(sum(base_preds))
        #print(sum(y))
        if choice == 'nll':
            y = test_f.y
            c = partial(tf.constant, dtype=dtype)
            #if net in ['normal', 'baseline_norm']:
            #    from  agg_learn.networks.normal_net import normal_likelihood
            #    log_bw_sq = 0.0 #TO BE IMPLEMENTED
            #    nll = normal_likelihood(c(y), c(base_preds), log_bw_sq, bag_size=test_f.bag_pop)
            #elif net in ['poisson', 'baseline_pois']:
            nll = tf.reduce_sum(tf.nn.log_poisson_loss(c(y), tf.log(c(base_preds)), 
                                compute_full_loss=True))
            base_value = nll.eval()/len(test_f)
        elif choice == 'mse':
            if true_y:
                y = test_f.true_y
            else:
                y = test_f.y
            if log_y:
                y[y <= 0] = 1.0
                y = np.log(y) 
                base_preds = np.log(base_preds)
            from sklearn.metrics import mean_squared_error
            base_value = mean_squared_error(y, base_preds)
        #print('bag_pop', test_f.bag_pop[:50])
        #print('base_y', 'log: {}'.format(log_y), base_preds[:50])
        #print('true_y', y[:50].tolist())
    else:
        raise ValueError('Please specify criterion as bag or indiv')
    return base_value

def eval_network(net, test_f, batch_pts, landmarks=None, batch_bags=np.inf, 
                 net_type=None, criterion='bag', gen_type=None, log_y=False,
                 link=None):
    if criterion in ['bag']:#, 'bag_mean']:
        preds = np.zeros_like(test_f.y)
    elif criterion in ['indiv']:#, 'indiv_mean']:
        preds = np.zeros(test_f.total_points)
    elif criterion in ['bag_nll', 'indiv_nll']:#, 'bag_nll_mean', 'indiv_nll_mean']:
        preds = 0
    i = 0
    if net_type == 'vb' and criterion == 'calibration':
        test_f.make_stacked()
        true_y = test_f.stacked_features[:,-2]
        d = net.feed_dict(test_f, test_f.y, landmarks)
        preds_mean = net.mu.eval(feed_dict=d)
        #print('check', net.Sigma_diag.eval(feed_dict=d)[:100])
        preds_sigma = np.sqrt(net.Sigma_diag.eval(feed_dict=d))
        #print(preds_sigma.tolist())
        quantile_calibrate = []
        # Compute calibration quantiles
        for quantile in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            if gen_type == 'poisson':
                if link == 'exp':
                    lower_q, upper_q = log_norm_quantile(preds_mean, preds_sigma, quantile, log=True)
                elif link == 'square':
                    mean_square = np.square(preds_mean)
                    sigma_square = np.square(preds_sigma)
                    nc_array = np.divide(mean_square, sigma_square)
                    # non-central chi square
                    ci = [list(ss.ncx2.interval(quantile, 1, nc_array[i], 
                                                scale=sigma_square[i])) for i in range(len(mean_square))]
                    lower_q, upper_q = np.array(ci)[:,0], np.array(ci)[:,1]
            elif gen_type == 'normal':
                #normal quantiles
                lower_q, upper_q = log_norm_quantile(preds_mean, preds_sigma, quantile, log=False)
            #print(((true_y < upper_q) & (lower_q < true_y)).tolist())
            calibration = np.mean((true_y < upper_q) & (lower_q < true_y))
            print('Calibration: {} Truth: {}'.format(calibration, quantile))
            quantile_calibrate.append(calibration)
        return quantile_calibrate
    else:
        for batch in loop_batches(test_f, max_pts=batch_pts, max_bags=batch_bags,
                                  stack=True, shuffle=False):
            d = net.feed_dict(batch, batch.y, landmarks)
            if criterion == 'bag':
                preds[i:i + len(batch)] = net.bag_y.eval(feed_dict=d)
                i += len(batch)
            elif criterion == 'indiv':
                preds[i:i + batch.total_points] = net.indiv.eval(feed_dict=d)
                i += batch.total_points
            elif criterion == 'bag_nll':
                preds = preds + net.bag_nll.eval(feed_dict=d)
            elif criterion == 'indiv_nll':
                preds = preds + net.indiv_nll.eval(feed_dict=d)
        if criterion in ['bag_nll']:
            preds = preds / len(test_f)
        elif criterion in ['indiv_nll']:
            preds = preds / test_f.total_points
        if criterion == 'bag' and log_y:
            preds[preds <= 0.0] = 1 
            preds = np.log(preds)
        return preds
