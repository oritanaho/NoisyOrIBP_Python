@Naho Orita (naho@umd.edu)

This is an implementation of Noisy-Or IBP (Gibbs sampling) in Python
based on the Matlab code by Frank Wood: 
http://www.robots.ox.ac.uk/~fwood/Code/ibp.zip

*** This code comes with no guarantees. ***
You will need to have the numpy and scipy packages installed.

Reference:
Wood et al. 2006. A Non-Parametric Bayesian Method for Inferring
Hidden Causes.

Data:
'toy_data' contains synthetic examples.
50 (data points) x 4 (ingredient features) matrix randomly assigned 0
or 1. Change it accordingly.

How to use:
> python no_ibp.py iterations alpha epsilon lambda p max_newK

:iterations: a number of iterations (say 1000-2000?)
:alpha, epsilon, lambda p: hyperparameters described in Wood et
al. (2006). Hyperparameters are not sampled in this code.
:max_newK: a maximum number of latent features

TODO:
Show results! You can print the resulting matrices Z and Y (line 375)
if you want.
