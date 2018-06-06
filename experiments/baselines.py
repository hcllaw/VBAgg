import os
from itertools import product
import subprocess
import sys

import numpy as np

# Obtain the environment variables to determine the parameters

# Setup we will like to grid search over
bag_size = np.arange(50, 400, 50)
data_seed = np.arange(0, 5)
gen_type = ['poisson']
learn_rate = [0.05, 0.01, 0.001]
rep_seed = np.arange(0, 3) 
reg_out = [1.0, 0.1, 0.0]
structure = [15, 30]
link = ['square', 'exp']
mean = 150
std = 50

vector_grid = list(product(data_seed, gen_type, learn_rate, rep_seed, reg_out, structure, link))
# take say first one
data_seed_c, gen_type_c, learn_rate_c, rep_seed_c, reg_out_c, structure_c, link_c = vector_grid[0]
data_seed_c = int(data_seed_c)
rep_seed_c = int(rep_seed_c)
structure_c = int(structure_c)

folder_path = 'path/to/save/to/baseline_results'

for bag_size_c in bag_size:
    file_path = os.path.join(folder_path, 'bag_{}_mean_{}_std_{}_gen_{}_net_baseline_link_{}_structure_{}_reg_NA_reg_out_{}_lr_{}_data_seed_{}_init_{}'.format(
                             bag_size_c, mean, std, gen_type_c, link_c, structure_c, reg_out_c, learn_rate_c, data_seed_c, rep_seed_c))
    # Save output and parameters to text file in the localhost node,
    # which is where the computation is performed.
    command = [
        "python", "train_test.py",
        'swiss',
        '--n-train', str(bag_size_c),
        '--n-estop', str(bag_size_c),
        '--n-val', str(bag_size_c),
        '--link', str(link_c),
        '--n-test', str(500),
        '--size-mean', str(mean),
        '--size-std', str(std),
        '--max-epochs', str(50),
        '--dim', str(18),
        '--size-type', 'neg-binom',
        '--dtype-single',
        '--batch-bags', str(int(bag_size_c/10)),
        '--eval-batch-bags', str(int(bag_size_c/10)),
        '--learning-rate', str(learn_rate_c),
        '--structure', str(structure_c),
        '--bw-scale', str(1.0),
        '--reg-out', str(reg_out_c),
        '--y-gen-type', gen_type_c,
        '--net-type', 'nn',
        '--tune', 'bag',
        '--estop-criterion', 'nll',
        '--data-noise', str(0.0),
        '--data-seed', str(data_seed_c),
        '--opt-seed', str(rep_seed_c),
        file_path
    ]
    cmd = subprocess.list2cmdline(command)
    print(cmd)
    os.system(cmd)


