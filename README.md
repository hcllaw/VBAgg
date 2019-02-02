# VBAgg
Python code (tested on 2.7) for aggregate output learning with Gaussian processes, the details are described in the following paper:

H. Law, D. Sejdinovic, E. Cameron, T. CD Lucas, S. Flaxman, K. Battle, K. Fukumizu, Variational Learning on Aggregate Outputs with Gaussian Processes, NeurIPS 2018 (https://arxiv.org/abs/1805.08463)

At the moment, the code is not optimised and many parts is not very clean. Hence, I would recommend to wait till April, where I will have instructions and also a cleaner version of this code.

Due to data confidentiality reasons, we do not provide the malaria data we used in the paper.

## Setup
To setup as a package, clone the repository and run
```
python setup.py develop
```
This package also requires TensorFlow (tested on v1.7.0) to be installed.

## Structure
The directory is organised as follows:
* __vb_agg_learn__: contains the main code, including toy data scripts
* __experiment__: contains the API code and experimental configuration code
