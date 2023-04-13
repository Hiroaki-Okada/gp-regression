import pdb

import math

import numpy as np
from scipy.stats import norm


# EI 関数
def ei(train_y, test_x, mean, variance, jitter=0.01):
    best_y = np.max(train_y)
    stdev = np.sqrt(variance) + 1e-6

    z = (mean - best_y - jitter) / stdev
    imp = mean - best_y - jitter

    ei = imp * norm.cdf(z) + stdev * norm.pdf(z)

    return ei


# PI 関数
def pi(train_y, test_x, mean, variance, jitter=0.01):
    best_y = np.max(train_y)
    stdev = np.sqrt(variance) + 1e-6

    z = (mean - best_y - jitter) / stdev
    cdf = norm.cdf(z)

    return cdf


# UCB 関数
def ucb(train_y, test_x, mean, variance, jitter=0.01, delta=0.5):
    stdev = np.sqrt(variance) + 1e-6

    dim = 1
    iters = len(train_y)

    beta = np.sqrt(2 * np.log(dim * (iters**2) * (math.pi**2) / (6*delta)))
    ucb = mean + beta * stdev

    return ucb
