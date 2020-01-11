# -*- coding: utf-8 -*-
"""
Created on Sat Jan 9 07:46:05 2016

Parallel IBP with warm start. Adapted from Ke Zhai's IBP code.

@author: Michael Zhang
"""
from __future__ import division
from past.builtins import xrange
import argparse
import time
import math
import numpy as np
import scipy.stats
import os
import datetime
from scipy.io import savemat
from mpi4py import MPI
np.seterr(divide='ignore')

class HybridIBPAll(object):

    def __init__(self, data, alpha=1.0, sigma_a=1.0, sigma_x=1.0,
                 A_prior=None, initial_Z=None, initial_A = None,
                 fname="synthetic",
                 alpha_hyper_parameter=(1.0,1.0),
                 sigma_a_hyper_parameter=(1.0,1.0),
                 sigma_x_hyper_parameter=(1.0,1.0), verbose=True,
                 iteration=1000, warm_start=True, L=5, 
                 always_on = False, test_prop = .1):
        """
        This is a parallel MCMC sampler for the linear Gaussian-Indian Buffet 
        Process model as used in Zhang, Dubey, Williamson (2015) and 
        Dubey, Zhang, Xing, Williamson (2020).
        
        @param data: NxD numpy array representing the observations
        @param alpha: positive float, concentration parameter in IBP
        @param sigma_a: positive float, noise parameter of the features
        @param sigma_x: positive float, noise parameter of the data
        @param A_prior: 1xD numpy array, prior on the features, defaults to 
                        vector of zeros
        @param initial_Z: NxK binary numpy array, must have same number of 
                            columns as instantiated features. Initial values of 
                            Z. Defaults to sampling from IBP prior
        @param initial_A: KxD numpy array, Initial values of A. Defaults to MAP
                            estimate of A.
        @param fname: string, file name of output file.
        @param alpha_hyper_parameter: size 2 tuple, hyperparameter values of 
                                        alpha
        @param sigma_a_hyper_parameter: size 2 tuple, hyperparameter values of 
                                        sigma_a
        @param sigma_x_hyper_parameter: size 2 tuple, hyperparameter values of 
                                        sigma_x
        @param verbose: boolean, if True, prints MCMC information at each 
                        synchronizatoin step
        @param iteration: int, number of total MCMC iterations
        @param warm_start: boolean, if True, allows all processors to introduce
                            features for beginning 1/8-th of total MCMC iterations 
        @param L: int, regularity of synchronization steps. Sampler will
                    trigger the syncrhonization step every L iterations.
        @always_on: boolean, if True then all processors will propose new 
                    features for all iterations. Assumed False if warm_star 
                    is True
        @test_prop: float between 0 and 1, proportion of cells in X to hold out
                    and impute for test-set log likelihood calculations.
        """
        self.comm = MPI.COMM_WORLD
#        self.particle_prob = particle_prob
        self._P = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.verbose = verbose
        self._sigma_a = sigma_a
        self._sigma_x = sigma_x
        self._alpha = alpha
        self.today = datetime.datetime.today()
        self.fname_head= str(fname)
        self.iters = int(iteration)
        self.warm_start = bool(warm_start)
        self.test_prop = test_prop
        assert(self.test_prop < 1. and self.test_prop > 0.)
        if self.warm_start:
            self.always_on = False
        else:
            self.always_on = bool(always_on)
        self._L = int(L)

        if self.rank == 0:
            self._X = self.center_data(data).astype(float);
            (self._N, self._D) = self._X.shape
            if self._D % 2 == 0:
                even_mask = tuple([0,1]*(self._D//2))
                odd_mask = tuple([1,0]*(self._D//2))
            else:
                even_mask = [0,1]*(self._D//2 + 1)
                even_mask = tuple(even_mask[:self._D])
                odd_mask = [1,0]*(self._D//2 + 1 )
                odd_mask = tuple(odd_mask[:self._D])
            idx_mask = np.array([even_mask,odd_mask] * (int(self._N * self.test_prop)/ int(2)))
            idx_N = idx_mask.shape[0]
            zeros_mask = np.zeros((self._N - idx_N, self._D))
            self.mask = np.vstack((idx_mask, zeros_mask)).astype(float)
            assert(self.mask.shape == self._X.shape)

            if initial_Z is None :
                # initialize Z from IBP(alpha)
                self._Z = self.initialize_Z();
            else:
                self._Z = initial_Z;
            self._Z = self._Z.astype(float)
            assert(self._Z.max() == 1 and self._Z.min() == 0);

            assert(self._Z.shape[0] == self._N);
            if A_prior is None:
                self._A_prior = np.zeros((1, self._D));
            else:
                self._A_prior = A_prior;
            assert(self._A_prior.shape == (1, self._D));

            self._K = self._Z.shape[1];
            self._K_plus = self._Z.shape[1];

            if initial_A is None:
                self._A = self.init_map_estimate_A();
            else:
                self._A = initial_A;
            assert(self._A.shape == (self._K, self._D));

            self._pi = np.random.beta(1.0, self._alpha, size = self._K)
            self.test_idx = self.mask.sum(axis=1).nonzero()[0]
            self.X_test = np.ma.array(self._X[self.test_idx,:], mask=1-idx_mask)

            self._X = np.array_split(self._X, self._P)
            self.mask = np.array_split(self.mask, self._P)
            self._Z_sum = self._Z.sum(axis=0)
            self._Z = np.array_split(self._Z, self._P)
            self._P_prime = np.random.choice(xrange(self._P), self._P)
            self.new_feat = np.array([self._K])

        else:
            self._X = None
            self._Z = None
            self._N = None
            self._D = None
            self._A = None
            self._A_prior = None
            self._K = None
            self._K_plus = None
            self._pi = None
            self._Z_sum = None
            self.test_idx = None
            self.X_test = None
            self._P_prime = None
            self.mask = None
            self.new_feat = None

#        self.error_trace = self.comm.bcast(self.error_trace)
        self._N = self.comm.bcast(self._N)
        self._D = self.comm.bcast(self._D)
        self._A = self.comm.bcast(self._A)
        self._pi = self.comm.bcast(self._pi)
        self._K = self.comm.bcast(self._K)
        self._K_plus = self.comm.bcast(self._K_plus)
        self._A_prior = self.comm.bcast(self._A_prior)
        self._K_star = 0
        self._Z_sum = self.comm.bcast(self._Z_sum)
        self._P_prime = self.comm.bcast(self._P_prime)


        self._X_local = self.comm.scatter(self._X)
        self._Z_local = self.comm.scatter(self._Z)
        self.mask_local = self.comm.scatter(self.mask)

        self._X_local_eval = np.ma.array(self._X_local, mask=1-self.mask_local)
        self._X_local = np.ma.array(self._X_local, mask=self.mask_local)

        assert(self._X_local.shape[0] == self._Z_local.shape[0])
        self._N_p,_ = self._X_local.shape
        assert(self._N_p > 0)


        if self.rank == 0:
            error = self._X_local - np.dot(self._Z_local, self._A )
            self.error_trace = np.trace(np.dot(error.T, error))
        else:
            self.error_trace = None

        self._A = self.comm.bcast(self._A)
        self.error_trace = self.comm.bcast(self.error_trace)
        assert(self._A.shape == (self._K, self._D));

        local_XZA = self._X_local - np.dot(self._Z_local, self._A)
        local_XZA_square = np.dot(local_XZA.T, local_XZA)
        self.error_local = local_XZA_square


        # initialize the hyper-parameter for sampling _sigma_x
        # a value of None is a gentle way to say "do not sampling _sigma_x"
        assert(sigma_x_hyper_parameter == None or type(sigma_x_hyper_parameter) == tuple);
        self._sigma_x_hyper_parameter = sigma_x_hyper_parameter;

        # initialize the hyper-parameter for sampling _sigma_a
        # a value of None is a gentle way to say "do not sampling _sigma_a"
        assert(sigma_a_hyper_parameter == None or type(sigma_a_hyper_parameter) == tuple);
        self._sigma_a_hyper_parameter = sigma_a_hyper_parameter;
        assert(alpha_hyper_parameter == None or type(alpha_hyper_parameter) == tuple);
        self._alpha_hyper_parameter = alpha_hyper_parameter;
        self._X = None
        self._Z = None
        self.mask = None

        self.local_map_time = []
        self.local_reduce_time = []

    def partition_tuple(self):
        base = [int(self._N / self._P) for k in range(self._P)]
        remainder = self._N % self._P
        assert(len(xrange(remainder)) <= len(base))
        if remainder:
            for r in xrange(remainder):
                base[r] += 1
        assert(sum(base) == self._N)
        assert(len(base) == self._P)
        return(tuple(base))

    def displacement(self, partition):
        if self._P > 1:
            displace = np.append([0], np.cumsum(partition[:-1])).astype(int)
        else:
            displace = [0]
        return(tuple(displace))

    def initialize_Z(self):
        Z = np.ones((0, 0));
        # initialize matrix Z recursively in IBP manner
        for i in xrange(1, self._N + 1):
            # sample existing features
            # Z.sum(axis=0)/i: compute the popularity of every dish, computes the probability of sampling that dish
            sample_dish = (np.random.uniform(0, 1, (1, Z.shape[1])) < (Z.sum(axis=0).astype(np.float) / i));
            # sample a value from the poisson distribution, defines the number of new features
            K_new = scipy.stats.poisson.rvs((self._alpha * 1.0 / i));
            # horizontally stack or append the new dishes to current object's observation vector, i.e., the vector Z_{n*}
            sample_dish = np.hstack((sample_dish, np.ones((1, K_new))));
            # append the matrix horizontally and then vertically to the Z matrix
            Z = np.hstack((Z, np.zeros((Z.shape[0], K_new))));
            Z = np.vstack((Z, sample_dish));

        assert(Z.shape[0] == self._N);
        Z = Z.astype(np.int);

        return Z


    """
    maximum a posterior estimation of matrix A
    todo: 2D-prior on A when initializing A matrix
    """
    def init_map_estimate_A(self):
        (mean, std_dev) = self.init_sufficient_statistics_A();
        assert(mean.shape == (self._K, self._D));

        return mean

    """
    compute the M matrix
    @param Z: default to None, if set, M matrix will be computed according to the passed in Z matrix
    """
    def compute_M(self, Z=None):
        if Z is None:
            Z = self._Z_local;

        K = Z.shape[1];
        M = np.linalg.inv(np.dot(Z.transpose(), Z) + (self._sigma_x / self._sigma_a) ** 2 * np.eye(K));
        return M

    """
    compute the mean and co-variance, i.e., sufficient statistics, of A
    @param observation_index: a list data type, recorded down the observation indices (column numbers) of A we want to compute
    """
    def init_sufficient_statistics_A(self):
        # compute M = (Z' * Z - (sigma_x^2) / (sigma_a^2) * I)^-1
        M = self.compute_M(self._Z);
        # compute the mean of the matrix A
        mean_A = np.dot(M, np.dot(self._Z.transpose(), self._X));
        # compute the co-variance of the matrix A
        std_dev_A = np.linalg.cholesky(self._sigma_x ** 2 * M).transpose();

        return (mean_A, std_dev_A)
    """
    center the data, i.e., subtract the mean
    """
    def center_data(self,data):
        (N, D) = data.shape;
        data = data - np.tile(data.mean(axis=0), (N, 1));
        return data

    def collpased_Zn(self, object_index):
        new_m = (self._Z_sum - self._Z_local[object_index, :]).astype(np.float);
        ucs_prob_z1 = np.log(self._pi)
        ucs_prob_z0 = np.log(1.0 - self._pi)
        log_prob_z1 = np.log(new_m / self._N);
        log_prob_z0 = np.log(1.0 - (new_m / self._N));
        singleton_features = [nk for nk in range(self._K) if self._Z_local[object_index, nk] != 0 and new_m[nk] == 0];
        non_singleton_features = [nk for nk in range(self._K) if nk not in singleton_features]
        K_star = np.arange(self._K_plus, self._K)
        order = np.random.permutation(self._K);
        for (feature_counter, feature_index) in enumerate(order):
            if feature_index in non_singleton_features:
                if feature_index >= self._K_plus:
                    self._Z_local[object_index, feature_index] = 0.;
                    prob_z0 = self.partial_collapsed_log_likelihood_X(X=self._X_local, Z=self._Z_local, K_star = K_star)
                    prob_z0 += log_prob_z0[feature_index];

                    self._Z_local[object_index, feature_index] = 1.;
                    prob_z1 = self.partial_collapsed_log_likelihood_X(X=self._X_local, Z=self._Z_local, K_star = K_star)
                    prob_z1 += log_prob_z1[feature_index]

                    prob = np.array([prob_z0, prob_z1])
                    max_p = np.max(prob)
                    log_sum_exp = max_p + np.log(np.sum(np.exp(prob - max_p)))
                    Znk_is_0 = prob_z0 - log_sum_exp
                    if np.log(np.random.random()) < Znk_is_0:
                        self._Z_local[object_index, feature_index] = 0;
                    else:
                        self._Z_local[object_index, feature_index] = 1;

                else:
                    self._Z_local[object_index, feature_index] = 0;
                    prob_z0 = self.log_likelihood_X(self._X_local[[object_index], :], self._Z_local[[object_index], :]);
                    prob_z0 += ucs_prob_z0[feature_index];

                    self._Z_local[object_index, feature_index] = 1;
                    prob_z1 = self.log_likelihood_X(self._X_local[[object_index], :], self._Z_local[[object_index], :]);
                    prob_z1 += ucs_prob_z1[feature_index]

                    prob = np.array([prob_z0, prob_z1])
                    max_p = np.max(prob)
                    log_sum_exp = max_p + np.log(np.sum(np.exp(prob - max_p)))
                    Znk_is_0 = prob_z0 - log_sum_exp
                    if np.log(np.random.random()) < Znk_is_0:
                        self._Z_local[object_index, feature_index] = 0;
                    else:
                        self._Z_local[object_index, feature_index] = 1;
        return singleton_features;

    def collapsed_MH(self, object_index, singleton_features):
        K_temp = np.random.poisson(self._alpha / self._N)

        if K_temp <= 0 and len(singleton_features) <= 0:
            return 0;

        K_star_old = np.arange(self._K_plus,self._K).astype(int)
        prob_old = self.partial_collapsed_log_likelihood_X(X=self._X_local, Z=self._Z_local, K_star = K_star_old)

        Z_new = np.hstack((self._Z_local, np.zeros((self._N_p, K_temp))));
        Z_new[[object_index], [xrange(-K_temp, 0)]] = 1;
        Z_new[[object_index], singleton_features] = 0;
        Z_new = Z_new.astype(int)

        if K_temp:
            K_star_new = np.arange(self._K_plus, K_temp + self._K).astype(int)
            assert(Z_new.shape[1] == K_temp + self._K)
        else:
            K_star_new = np.arange(self._K_plus,self._K).astype(int)

        prob_new = self.partial_collapsed_log_likelihood_X(X=self._X_local, Z=Z_new, K_star=K_star_new)
        prob = np.array([prob_old, prob_new])
        max_p = np.max(prob)
        log_sum_exp = max_p + np.log(np.sum(np.exp(prob - max_p)))
        accept_new =  prob_new - log_sum_exp

        if np.log(np.random.random()) < accept_new:
            self._K_star += K_temp
            self._K += K_temp
            assert(Z_new.shape[1] == self._K_star + self._K_plus)
            self._Z_local = np.copy(Z_new)
            if K_temp:
                self._Z_sum = np.append(self._Z_sum, np.ones(K_temp))
                self._A = np.vstack((self._A, np.zeros((K_temp, self._D))))
                assert(self._A.shape[0] == self._K)
                assert(self._A.shape[1] == self._D)
#                self._pi = new_pi
            return 1;

        else:
            return 0;

    def map_step(self):
        start_time = time.time()
        N_p_order = np.random.permutation(self._N_p)
        for (object_counter, object_index) in enumerate(N_p_order):
            if self.rank in self._P_prime:
                singleton = self.collpased_Zn(object_index)
                self.collapsed_MH(object_index, singleton)
            else:
                singleton = self.collpased_Zn(object_index)
        map_time = time.time() - start_time
        self.local_map_time.append(map_time)

    def new_Z_idx(self, K_star_list):
        assert(len(K_star_list) == self._P)
        new_idx = []
        lower = self._K_plus
        for p in xrange(self._P):
            if K_star_list[p] == 0:
                new_idx.append([])
            else:
                new_idx.append(range(lower, lower+K_star_list[p]))
                lower = lower+K_star_list[p]
        return(new_idx)

    def new_Z(self):
        K_star_list = self.comm.gather(self._K_star)
        if self.rank == 0:
            sum_K_star = sum(K_star_list)
            new_K = self._K_plus + sum_K_star
            self._K = new_K
            new_idx = self.new_Z_idx(K_star_list)
            self.new_feat = np.hstack((self.new_feat, sum_K_star))
        else:
            K_star_list = None
            sum_K_star = None
            new_idx = None
            new_K = None

        new_idx = self.comm.bcast(new_idx)
        self._K = self.comm.bcast(new_K)
        A = np.zeros((self._K, self._D))
        A[xrange(self._K_plus)] = np.copy(self._A[xrange(self._K_plus)])
        self._A = A

        assert(len(new_idx) == self._P)
        new_pi = np.zeros(self._K)
        new_pi[xrange(self._K_plus)] = np.copy(self._pi[xrange(self._K_plus)])
        self._pi = new_pi

        Z_mask = np.append(np.arange(self._K_plus), new_idx[self.rank]).astype(int)
        new_Z_local = np.zeros((self._N_p, self._K))
        new_Z_local[:,Z_mask] = np.copy(self._Z_local)
        self._Z_local = new_Z_local

    def reduce_step(self, num_P):
        start_time = time.time()
        local_sum = self._Z_local.sum(axis=0)

        local_M_inv = np.dot(self._Z_local.T, self._Z_local) # + (self._sigma_x / self._sigma_a) ** 2 * np.eye(self._K);
        M = np.linalg.inv(self.comm.allreduce(local_M_inv) + (self._sigma_x / self._sigma_a) ** 2 * np.eye(self._K))

        local_MZX = np.dot(M, np.ma.dot(self._Z_local.T, self._X_local))
        local_XZA = self._X_local - np.ma.dot(self._Z_local, self._A)
        local_XZA_square = np.ma.dot(local_XZA.T, local_XZA)
        self.error_local = local_XZA_square

        self._Z_sum = self.comm.allreduce(local_sum)
        MZX = self.comm.reduce(local_MZX)
        X_minus_ZA = self.comm.reduce(local_XZA_square)

        if self.rank == 0:
            self._pi = np.random.beta(self._Z_sum, 1. + self._N - self._Z_sum)
            self._alpha = self.sample_alpha()

            A_sigma_chol = np.linalg.cholesky(self._sigma_x**2 * M)
            self._A = MZX + np.dot(A_sigma_chol, np.random.normal(size = (self._K, self._D)))

            self.error_trace = np.ma.trace(X_minus_ZA)
            gamma_x_a = self._sigma_x_hyper_parameter[0] + (.5 * self._N * self._D)
            gamma_x_b = 1. / (self._sigma_x_hyper_parameter[1] + (.5 * self.error_trace))
            self._sigma_x = np.sqrt(1./ np.random.gamma(gamma_x_a, gamma_x_b))

            A_minus_prior = self._A - self._A_prior
            A_square = np.trace(np.dot(A_minus_prior.T, A_minus_prior))
            gamma_a_a = self._sigma_a_hyper_parameter[0] + (.5 * self._K * self._D)
            gamma_a_b = 1. / (self._sigma_a_hyper_parameter[1] + (.5 * A_square))
            self._sigma_a = np.sqrt(1. / np.random.gamma(gamma_a_a, gamma_a_b))
            self._P_prime = np.random.choice(xrange(self._P), num_P)
        else:
            self.error_trace = None

        self.error_trace = self.comm.bcast(self.error_trace)
        self._A = self.comm.bcast(self._A)
        self._P_prime = self.comm.bcast(self._P_prime)
        self._sigma_a = self.comm.bcast(self._sigma_a)
        self._sigma_x = self.comm.bcast(self._sigma_x)
        self._pi = self.comm.bcast(self._pi)
        self._alpha = self.comm.bcast(self._alpha)
        reduce_time = time.time() - start_time
        self.local_reduce_time.append(reduce_time)

    """
    remove the empty column in matrix Z and the corresponding feature in A
    """
    def regularize_matrices(self):

        Z_sum = self.comm.reduce(self._Z_local.sum(axis=0))
        if self.rank == 0:
            indices = np.nonzero(Z_sum == 0)[0];
            new_idx = np.sort(np.array([k for k in range(self._K) if k not in indices])).astype(int)
            self._K = new_idx.size
            self._A = self._A[new_idx, :];
        else:
            Z_sum = None
            indices = None
            new_idx = None

        new_idx = self.comm.bcast(new_idx)
        self._K = self.comm.bcast(self._K)
        self._A = self.comm.bcast(self._A)
        self._Z_local = self._Z_local[:, new_idx];
        self._pi = self._pi[new_idx]
        assert(self._pi.size == self._K)
        assert(self._Z_local.shape[1] == self._A.shape[0])
        self._K_plus = self._K
        self._K_star = 0
        assert(self._Z_local.shape == (self._N_p, self._K));
        assert(self._A.shape == (self._K, self._D));

    """
    compute the log-likelihood of the data X
    @param X: a 2-D np array
    @param Z: a 2-D np boolean array
    @param A: a 2-D np array, integrate A out if it is set to None
    """

    def log_likelihood_X(self, X=None, Z=None, A=None):
        if A is None:
            A = self._A;
        if Z is None:
            Z = self._Z_local;
        if X is None:
            X = self._X_local;

        assert(X.shape[0] == Z.shape[0]);
        (N, D) = X.shape;
        (N, K) = Z.shape;
        assert(A.shape == (K, D));

        log_likelihood = X - np.ma.dot(Z, A);

        (row, column) = log_likelihood.shape;
        if row > column:
            log_likelihood = np.ma.trace(np.ma.dot(log_likelihood.T, log_likelihood));
        else:
            log_likelihood = np.ma.trace(np.ma.dot(log_likelihood, log_likelihood.T));

        log_likelihood = -0.5 * log_likelihood / np.power(self._sigma_x, 2);
        log_likelihood -= N * D * 0.5 * np.log(2 * np.pi * np.power(self._sigma_x, 2));

        return log_likelihood

    """
    compute the log-likelihood of A
    """
    def log_likelihood_A(self):
        log_likelihood = -0.5 * self._K * self._D * np.log(2 * np.pi * self._sigma_a * self._sigma_a);
        #for k in range(self._K):
        #    A_prior[k, :] = self._mean_a[0, :];
        A_prior = np.tile(self._A_prior, (self._K, 1))
        log_likelihood -= np.trace(np.dot((self._A - A_prior).transpose(), (self._A - A_prior))) * 0.5 / (self._sigma_a ** 2);

        return log_likelihood;

    """
    compute the log-likelihood of the Z matrix.
    """
    def log_likelihood_Z(self):
        # compute {K_+} \log{\alpha} - \alpha * H_N, where H_N = \sum_{j=1}^N 1/j
        H_N = np.array([range(self._N_p)]) + 1.0;
        H_N = np.sum(1.0 / H_N);
        log_likelihood = self._K * np.log(self._alpha) - self._alpha * H_N;

        # compute the \sum_{h=1}^{2^N-1} \log{K_h!}
        Z_h = np.sum(self._Z_local, axis=0).astype(np.int);
        Z_h = list(Z_h);
        for k_h in set(Z_h):
            log_factorial = math.log(math.factorial(Z_h.count(k_h)));
            log_likelihood -= log_factorial

        # compute the \sum_{k=1}^{K_+} \frac{(N-m_k)! (m_k-1)!}{N!}
        for k in xrange(self._K):
            m_k = Z_h[k];
            temp_var = 1.0;
            if m_k - 1 < self._N - m_k:
                for k_prime in range(self._N_p - m_k + 1, self._N_p + 1):
                    if m_k != 1:
                        m_k -= 1;

                    temp_var /= k_prime;
                    temp_var *= m_k;
            else:
                n_m_k = self._N_p - m_k;
                for k_prime in range(m_k, self._N_p + 1):
                    temp_var /= k_prime;
                    temp_var += n_m_k;
                    if n_m_k != 1:
                        n_m_k -= 1;

            log_likelihood += np.log(temp_var);

        return log_likelihood

    def collapsed_log_likelihood_X(self, Z=None, X=None, test=False):

        if Z is None:
            Z = self._Z_local;

        if X is None:
            X = self._X_local

        M = self.compute_M(Z)
        log_det_M = np.log(np.linalg.det(M));

        assert(X.shape[0] == Z.shape[0]);
        (N, D) = X.shape;
        (N, K) = Z.shape;
        assert(M.shape == (K, K));

        # we are collapsing A out, i.e., compute the log likelihood p(X | Z)
        # be careful that M passed in should include the inverse.
        log_likelihood = np.eye(N) - np.dot(np.dot(Z, M), Z.transpose());
        log_likelihood = -0.5 / (self._sigma_x ** 2) * np.ma.trace(np.ma.dot(np.ma.dot(X.T, log_likelihood), X));
        log_likelihood -= D * (N - K) * np.log(self._sigma_x) + K * D * np.log(self._sigma_a);
        log_likelihood += 0.5 * D * log_det_M;
        log_likelihood -= 0.5 * N * D * np.log(2 * np.pi);
        return(log_likelihood)


    """
    compute the log-likelihood of the model
    """
    def log_likelihood_model(self, X=None):
        return self.log_likelihood_X(X=X) + self.log_likelihood_A() + self.log_likelihood_Z();

    """
    sample alpha from conjugate posterior
    """
    def sample_alpha(self):
        assert(self._alpha_hyper_parameter != None);
        assert(type(self._alpha_hyper_parameter) == tuple);

        (alpha_hyper_a, alpha_hyper_b) = self._alpha_hyper_parameter;

        posterior_shape = alpha_hyper_a + self._K;
        H_N = np.array([range(self._N)]) + 1.0;
        H_N = np.sum(1.0 / H_N);
        posterior_scale = 1.0 / (alpha_hyper_b + H_N);

        alpha_new = scipy.stats.gamma.rvs(posterior_shape, scale=posterior_scale);

        return alpha_new;

    def partial_collapsed_log_likelihood_X(self, X=None, Z=None, A=None, K_star=None):
        if X is None:
            X = self._X_local
        if Z is None:
            Z = self._Z_local
        if A is None:
            A = self._A

        if K_star is None:
            if self._K_star:
                K_star = np.arange(self._K_plus, self._K)
                K_star_size = K_star.size
            else:
                K_star = []
                K_star_size = 0
        else:
            K_star_size = K_star.size

        K_plus = np.arange(self._K_plus)
        Z_plus = Z[:, K_plus]
        Z_star = Z[:, K_star]
        A_plus = A[K_plus,:]
        X = X - np.ma.dot(Z_plus, A_plus)

        assert(X.shape[0] == Z.shape[0]);
        (N, D) = X.shape;
        (N, K) = Z.shape
        N = self._N
        new_error_trace = self.error_trace - np.trace(self.error_local) + np.trace(np.dot(X.T,X)) # take out local errors from last global step and add in current errors
        if K_star_size:
            M = self.compute_M(Z_star)
            log_det_M = np.log(np.linalg.det(M))
            log_likelihood = np.ma.dot(np.ma.dot(Z_star, M), Z_star.transpose())
            log_likelihood = np.ma.dot(np.ma.dot(X.T, log_likelihood),X)
            log_likelihood = -0.5 / (self._sigma_x ** 2) * (new_error_trace - np.trace(log_likelihood))
            log_likelihood -= (D * (N - K_star_size) * np.log(self._sigma_x)) + (K_star_size * D * np.log(self._sigma_a));
            log_likelihood += 0.5 * D * log_det_M;
            log_likelihood -= 0.5 * N * D * np.log(2 * np.pi);

        else:
            log_likelihood = -0.5 / (self._sigma_x ** 2) * new_error_trace
            log_likelihood -= N * D * 0.5 * np.log(2 * np.pi * np.power(self._sigma_x, 2));

        return log_likelihood

    def sample(self):

        if self.rank==0:
            self.likelihood_iteration = np.empty((1,2))
            self.feature_count = [[0,self._K]]
            start_time = time.time()
            self._A_list = []
        else:
            self.likelihood_iteration = None
            start_time = None
            self.feature_count = None
            self._A_list = None

        counter = 0
        for i in xrange(self.iters):
            self.map_step()
            if i % self._L == 0 or iter == max(xrange(self.iters)):
                self.comm.Barrier()
                if self.always_on:
                    proc_num = self._P
                elif self.warm_start:    
                    # TODO: change warm start proportion into a parameter
                        if i <= self.iters//8: # for first 1/8th of iterations run warm start                            
                            proc_num = self._P
                        else:
                            proc_num = 1
                else:
                    proc_num = 1
                assert(proc_num > 0)
                counter += 1
                self.new_Z()
                self.regularize_matrices()
                self.reduce_step(num_P = proc_num)
                gather_Z = self.comm.gather(self._Z_local)
                if self.rank == 0:
                    gather_Z = np.vstack(gather_Z).reshape(self._N, self._K)[self.test_idx,:]
                    total_likelihood = self.collapsed_log_likelihood_X(X=self.X_test, Z = gather_Z)
                    self._A_list.append(np.array(self._A))
                    self.feature_count.append([i,self._K])
                    elapsed = time.time()-start_time
                    if self.warm_start:
                        anneal_type = "warm_start"
                    else:
                        anneal_type="cold_start"
                    self.likelihood_iteration = np.vstack((self.likelihood_iteration, (elapsed, total_likelihood)))
                    save_dict = {'Z_count':self.feature_count, 'likelihood':self.likelihood_iteration, 'features':self._A_list}
                    fname =  self.fname_head + "_hybridIBP" + "_P" + str(self._P) + "_" + anneal_type + ".mat"
                    savemat(os.path.abspath(fname),save_dict)
                    if self.verbose:
                        print("it: %i\tK: %i\tlikelihood: %f" % (i, self._K, total_likelihood));
                        print("alpha: %f\tsigma_a: %f\tsigma_x: %f\ttime: %f" % (self._alpha, self._sigma_a, self._sigma_x, elapsed));
                        print("feat. count: %s\tprocessors: %i\tprocessors on: %i" % (self._Z_sum, self._P,proc_num))

if __name__ == "__main__":
    base_seed =8889
    np.random.seed(base_seed)
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--data',
                        metavar='filename',
                        help='Data for model, must be np N*D array')
    parser.add_argument('--iters',
                        metavar='int', type=int, default=5000,
                        help='number of iteraitons, default 5000')
    parser.add_argument('--L',
                        metavar='int', type=int, default=5,
                        help='number of sub-iteraitons per iterations, default 5')
    parser.add_argument('--annealing',
                        action="store_true",
                        help='Flag turns on warm start')
    parser.add_argument('--allProc',
                        action="store_true",
                        help='Flag turns on all processor proposal')
    parser.add_argument('--initialZ',
                        metavar='filename', default=None,
                        help='Path and filename for initial values for Z, defaults to None')
    parser.add_argument('--fname', default="synthetic",
                        metavar='filename',
                        help='Filename header for output file')

    parser.set_defaults(data = os.path.abspath("../data/cambridge_10k.npy"))

    args = parser.parse_args()
    X=np.load(os.path.abspath(args.data))
    if args.initialZ is not None:
        Z = np.load(os.path.abspath(args.initialZ))
    else:
        Z = None

    hybrid = HybridIBPAll(data=X, initial_Z = Z,
                          iteration=args.iters, L=args.L,
                          warm_start=args.annealing, fname=args.fname,
                          always_on=args.allProc)
    hybrid.sample()
