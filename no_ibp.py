"""
@Naho Orita (naho@umd.edu)

This is an implementation of Noisy-Or IBP in Python
based on the code by Frank Wood: 
http://www.robots.ox.ac.uk/~fwood/Code/ibp.zip

"""

import sys
import time
from math import log
from random import randint

import numpy as np
import scipy
import scipy.stats
from scipy.special import gammaln


def run_sampler(iterations, X, alpha, epsilon, lamb, p, max_newK):
    """
    Run a number of Gibbs sampling iterations.
    :param iterations: The number of iterations to run the sampler.
    :params alpha, epsilon, lamb, p: hyperparameters. These are not
    sampled in this code.
    """
    
    lhoods = []
    Z, Y = initialize_ZY(X, alpha, max_newK)

    for iter in xrange(1, iterations+1):
        
        processing_time = time.time()
        
        Z, Y = sample_Z(X, Z, Y, alpha, epsilon, lamb, p, max_newK)
        Y = sample_Y(X, Z, Y, alpha, epsilon, lamb, p)
        Z, Y = sort_ZY(Z, Y)
        Z, Y = remove_empties_ZY(Z, Y)

        lhood = log_lhood(X, Z, Y, alpha, epsilon, lamb)        
        
        processing_time = time.time() - processing_time        

        print("iteration %d finished in %d seconds with log-likelihood %g"
              % (iter, processing_time, lhood))
            
    return Z, Y

    
def initialize_ZY(X, alpha, newK):
    """
    Initialize X, Z, Y
    :param X: observing matrix X, array
    """

    # matrix X
    (N, T) = X.shape
        
    # initial matrix Z
    initial_Z = initialize_Z(N, alpha, newK)
    assert(initial_Z.shape[0] == N)
    
    K = initial_Z.shape[1]

    # initial matrix Y
    Y = np.eye(K, T)

    return initial_Z, Y

        
def initialize_Z(N, alpha, newK):
    """
    Initialize matrix Z according to IBP.
    Copied from from Ke Zhai's code
    https://github.com/kzhai/PyNPB/blob/master/src/ibp/gs.py    
    """
    Z = np.ones((0, 0));
    # initialize matrix Z recursively in IBP manner
    for i in xrange(1, N + 1):
        # sample existing features
        # Z.sum(axis=0)/i: compute the popularity of every dish,
        # computes the probability of sampling that dish 
        sample_dish = (np.random.uniform(0, 1, (1, Z.shape[1])) <
                       (Z.sum(axis=0).astype(np.float) / i)); 
        # sample a value from the poisson distribution, defines the
        # number of new features 
        K_new = scipy.stats.poisson.rvs((alpha * 1.0 / i)); 
        # horizontally stack or append the new dishes to current
        # object's observation vector, i.e., the vector Z_{n*} 
        sample_dish = np.hstack((sample_dish, np.ones((1, newK)))); 
        # append the matrix horizontally and then vertically to the Z
        # matrix 
        Z = np.hstack((Z, np.zeros((Z.shape[0], newK))));
        Z = np.vstack((Z, sample_dish));
            
    assert(Z.shape[0] == N);
    Z = Z.astype(np.int);
        
    return Z

    
def log_lhood(X, Z, Y, a, ep, lamb):
    """
    Return the joint probability of X and Z.
    """
    
    K = Z.shape[1]
    N, T = X.shape
    
    # p(X)
    ZY = np.dot(Z,Y)        
    log_pX = 0
    log_pX = log_pX + np.sum(X * np.log(1 - ((1 - lamb) ** ZY) * (1 - ep)))
    log_pX = log_pX + np.sum((1 - X) * np.log(((1 - lamb) ** ZY) * (1 - ep))) 
    
    # p(Z)
    HN = 0
    for n in range(1, N+1):
        HN += 1.0/n
    m = Z.sum(axis=0)
    log_pZ = (K * np.log(a) - (a * HN)) + np.sum(gammaln(m) +
                                                 gammaln(N - m + 1) - gammaln(N + 1)) 

    return log_pZ + log_pX


def sample_Z(X, Z, Y, a, ep, lamb, p, max_newK):
    """
    Sample matrix Z
    1. sample individual cells in Z
    2. sample new columns in Zi
    Return new Z
    """

    K = Z.shape[1]
    N, T = X.shape
    assert(Z.shape[0] == N)

    for i in range(N):
        for k in range(K):
            Z = sample_individual_z(X, Z, Y, a, ep, lamb, p, K, N, T, i, k)
            
        Z, Y = sample_new_columns(X, Z, Y, a, ep, lamb, p, K, N, T, i,
                                  max_newK)

        m = (Z.sum(axis=0))
        non_empty_columns = [k for k in range(K) if m[k] > 0]
        Z = Z[:,non_empty_columns]
        Y = Y[non_empty_columns,:]
        K = Z.shape[1]

    return Z, Y

    
def sample_individual_z(X, Z, Y, alpha, ep, lamb, p, K, N, T, i, k):
    """
    Sample each entry in Z according to Eq (11).
    Return new Z
    """

    p_Zik_probs = {}

    # compute m_{-i,k}
    m_k = (Z.sum(axis=0) - Z[i,:])[k]

    # sample Zik if m_{-i,k} > 0
    if m_k > 0:
        for a in [0,1]:
            if a == 0:
                prior = log(1 - (m_k / float(N)))
            else:
                prior = log(m_k / float(N))

            Z[i,k] = a
            e = np.dot(Z[i,range(K)], Y[range(K),:])
            lhood = np.sum(X[i,:] * np.log(1 - ((1 - lamb) ** e) * (1 - ep)) + \
                               (1 - X[i,:]) * np.log((1 - lamb) ** e) * (1 - ep))

            p_Zik_probs[a] = prior + lhood

        # choose Z[i,k] = 0 or Z[i,k] = 1
        current = 1.0 / (1 + np.exp( -(p_Zik_probs[0] - p_Zik_probs[1])))
        rand = np.random.uniform()
        if current  > rand:
            Z[i,k] = 0
        else:
            Z[i,k] = 1            
            
    return Z


def sample_new_columns(X, Z, Y, alpha, ep, lamb, p, K, N, T, i, max_newK):
    """
    1. Sample new columns in Zi
    2. Construct new columns in Zi
    3. Construct corresponding new rows in Yk
    4. Sample new Yj,t 
    Return new Z and Y
    """
    
    # Sample new columns in Zi
    
    # compute m_{-i}
    m = (Z.sum(axis=0) - Z[i,:])
    empty_columns = [k for k in range(K) if m[k] == 0]
    # if no dish k is chosen by other customers, costomer i also
    # doesn't choose dish k 
    Z[i,empty_columns] = 0

    # compute probabilities for all possible new ks
    newK_probs = []
    e = np.dot(Z[i,range(K)], Y[range(K),:])
    # create indices of ones in Xi and zeros in Xi
    one_inds = [t for t in range(T) if X[i,t] == 1]
    zero_inds = [t for t in range(T) if t not in one_inds]
    # eta = (1 - lambda) ** np.dot(Z[i,1:K], Y[1:K,t]) in Eq(15)
    eta_one = np.power((1 - lamb), e[one_inds])
    eta_zero = np.power((1 - lamb), e[zero_inds])    

    for newk in range(1, max_newK + 1):
        lhood_XiT = 0
        lhood_XiT = np.sum(np.log(1 - (1 - ep) * eta_one * ((1 - lamb * p) ** newk)))
        lhood_XiT = lhood_XiT + np.sum(np.log((1 - ep) * eta_zero * ((1 - lamb * p) ** newk)))
        prob_newKi = lhood_XiT - alpha / N + (K + newk) * log(alpha / N) - gammaln(K + newk + 1)
        newK_probs.append(prob_newKi)

    # sample new k
    pdf = map(lambda x: np.exp(x - max(newK_probs)), newK_probs)
    normalized_pdf = map(lambda x: x / np.sum(pdf), pdf)
    cdf = pdf[0]
    cdfr = np.random.uniform()
    newK = 0
    ii = 0

    while cdf < cdfr:
        ii = ii + 1
        cdf = cdf + pdf[ii]
        newK = newK + 1

    if newK > 0:
        # construct new Z
        new_Z = np.hstack((Z, np.zeros((N, newK))))
        new_Z[i, xrange(-newK, 0)] = 1
        Z = new_Z

        # construct new Y
        new_Y = np.vstack((Y, np.zeros((newK, T))))

        # sample new Y_j,t: new values of Y are drawn from their
        # posterior dist given Z. 
        # I do not understand why Eq(12) can be the following. I just
        # translated Wood's code (sampZ.mat). 
        e_newY = np.tile(e, (newK+1, 1)) + np.tile(np.arange(newK+1).transpose(), (1, T))
        newY_probs = np.power(((1 - ep) * (1 - lamb)), e_newY)
        newY_probs[:,one_inds] = 1 - newprobs[:,one_inds]

        e_newY_prior = (n_choose_kv(newK) * np.power(p, np.arange(newK+1)) \
                            * np.power(1 - p, (newK - np.arange(newK+1)))).transpose()
        newY_prior = np.tile(e_newY_prior, (1, T))
        newY_probs = newY_probs * newY_prior 

        newY_probs = newY_probs / np.tile(np.sum(newY_probs), (newK+1, 1))
        newY_probs = np.cumsum(newY_probs)

        for j in range(T):
            rand = np.random.uniform()
            bigger_than_rand = [k for k in range(K) if rand < newY_probs[k,j]]
            m = min(bigger_than_rand)
            new_Y[xrange(-m, 0), j] = 1

        Y = new_Y

    return Z, Y


def n_choose_kv(newK):
    """
    Use this function for sampling Y_j,t.
    """
    values = np.zeros((1,newK+1))
    ks = np.arange(newK+1)
    
    for i in range(newK+1):
        values[i] = scipy.misc.comb(newK, ks[i])

    return values
        

def sample_Y(X, Z, Y, a, ep, lamb, p):
    """
    Sample entries in Y matrix.
    """

    K = Z.shape[1]
    N, T = X.shape
    assert(Z.shape[0] == N)

    # create empty Y matrices to put the probabilities when a = 0 and
    # a = 1 
    pY_a0 = np.zeros((K, T))
    pY_a1 = np.zeros((K, T))

    for t in range(T):
        for k in range(K):
            for a in [0,1]:
                Y[k,t] = a
                Y_prior = a * log(p) + (1 - a) * log(1 - p)
                e = np.dot(Z[:,:], Y[:,t])
                lhood = np.sum(X[:,t] * np.log(1 - ((1 - lamb) ** e) * (1 - ep)) + \
                               (1 - X[:,t]) * np.log((1 - lamb) ** e) * (1 - ep))
                p_Ykt = Y_prior + lhood

                if a == 0:
                    pY_a0[k,t] = p_Ykt
                else:
                    pY_a1[k,t] = p_Ykt      

            # choose Y[k,t] = 0 or 1                    
            current = 1.0 / (1 + np.exp( -(pY_a0[k,t] - pY_a1[k,t])))
            rand = np.random.uniform()
            if current  > rand:
                Y[k,t] = 0
            else:
                Y[k,t] = 1            
            
    return Y


def sort_ZY(Z, Y):

    N = Z.shape[0]    
    i = np.arange(N-1, -1, -1)
    p = np.power(2, i)
    sv = np.dot(p, Z)
    
    sorted_ind = np.argsort(sv.ravel(), axis=0)[::-1]
    new_Z = Z[:,sorted_ind]
    new_Y = Y[sorted_ind,:]

    return new_Z, new_Y


def remove_empties_ZY(Z, Y):    

    K = Z.shape[1]    
    m = Z.sum(axis=0)
    non_zeros = [k for k in range(K) if m[k] != 0]
    new_Z = Z[:,non_zeros]
    new_Y = Y[non_zeros,:]

    return new_Z, new_Y


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print "USAGE: python no_ibp.py iterations alpha epsilon lamb p max_newK"
        print "Try -> python no_ibp.py 1000 1.0 0.01 0.9 0.1 5"
            
    else:
        iterations = sys.argv[1]
        iterations = int(iterations)
        alpha = float(sys.argv[2])
        epsilon = float(sys.argv[3])
        lamb = float(sys.argv[4])
        p = float(sys.argv[5])
        max_newK = int(sys.argv[6])

# toy data: 50x4 matrix randomly assigned 0 or 1
X = np.array([[randint(0,1) for i in range(4)] for i in range(50)])

Z, Y = run_sampler(iterations, X, alpha, epsilon, lamb, p, max_newK)

print "TODO: plot results!"

