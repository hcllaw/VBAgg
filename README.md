# VBAgg
Python code (tested on 2.7) for aggregate output learning with Gaussian processes, the details are described in the following paper:

H. Law, D. Sejdinovic, E. Cameron, T. CD Lucas, S. Flaxman, K. Battle, K. Fukumizu, Variational Learning on Aggregate Outputs with Gaussian Processes, 2018 (https://arxiv.org/abs/1805.08463)

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

## Main API
The main API can be found in `/experiment`, where:
* `train_test.py`: contains API code for the different options available.
For example:
```
python train_test.py swiss --net-type vb /folder/to/save/to
```
would run the VBAgg algorithm and save results to ```/folder/to/save/to```, while using the default options. Experimental code for various models is found in ```/experiment``` and can be ran directly. 

Just a disclaimer that the code is not completely optimised, and some parts can be a little more clean. I will update it more in the recent future.
