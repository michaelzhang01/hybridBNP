# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 09:59:20 2016

Beta Bernoulli model--completely uncollapsed, parallel code

@author: Michael Zhang
"""

import pdb
import argparse
#import joblib
import numpy
import os
import time
import datetime
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.io import savemat


class BetaBernoulli(object):

    def __init__(self, data, alpha=1.0, sigma_a=1.0, sigma_x=1.0,
                 A_prior=None, initial_Z=None, initial_A = None,
                 alpha_hyper_parameter=(1.0,1.0),
                 sigma_a_hyper_parameter=(1.0,1.0),
                 sigma_x_hyper_parameter=(1.0,1.0), verbose=True,
                 K=4):
        self.comm = MPI.COMM_WORLD
#        self.particle_prob = particle_prob
        self._P = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.verbose = verbose
        self._sigma_a = sigma_a
        self._sigma_x = sigma_x
        self._alpha = alpha
        self._K = K
        self.today = datetime.datetime.today()

        if self.rank == 0:
            self._X = self.center_data(data).astype(float);
            (self._N, self._D) = self._X.shape
#            assert(self._D % 2 == 0)
#            pdb.set_trace()
            if self._D % 2 == 0:
                even_mask = tuple([0,1]*(self._D//2))
                odd_mask = tuple([1,0]*(self._D//2))
            else:
                even_mask = [0,1]*(self._D//2 + 1)
                even_mask = tuple(even_mask[:self._D])
                odd_mask = [1,0]*(self._D//2 + 1 )
                odd_mask = tuple(odd_mask[:self._D])
            idx_mask = numpy.array([even_mask,odd_mask] * (int(self._N * .1)/ int(2)))
            zeros_mask = numpy.zeros((self._N - int(self._N * .1), self._D))
            self.mask = numpy.vstack((idx_mask, zeros_mask)).astype(float)
            assert(self.mask.shape == self._X.shape)

            if(initial_Z is None):
                # initialize Z from IBP(alpha)
                self._Z = self.initialize_Z();
            else:
                self._Z = initial_Z;
                self._K = self._Z.shape[1];

            self._Z = self._Z.astype(float)
            assert(self._Z.max() == 1 and self._Z.min() == 0);

            assert(self._Z.shape[0] == self._N);
            if A_prior is None:
                self._A_prior = numpy.zeros((1, self._D));
            else:
                self._A_prior = A_prior;
            assert(self._A_prior.shape == (1, self._D));

            self._K = self._Z.shape[1];
#            self._K_plus = self._Z.shape[1];

            if initial_A != None:
                # this will replace the A matrix generated in the super class.
                self._A = initial_A;
            else:
                self._A = self.init_map_estimate_A();
            assert(self._A.shape == (self._K, self._D));

            self._pi = numpy.random.beta(1.0, self._alpha, size = self._K)
            self.test_idx = self.mask.sum(axis=1).nonzero()[0]
            self.X_test = numpy.ma.array(self._X[self.test_idx,:], mask=1-idx_mask)
#            self.mask_test = idx_mask
#            assert(self.X_train.shape == self.mask_train)
            self._X = self._X.flatten()
#            self.mask = self.mask.flatten()
            self._Z_sum = self._Z.sum(axis=0)
            self._Z = self._Z.flatten()
#            self._P_prime = numpy.random.choice(xrange(self._P), self._P)
#            self.new_feat = numpy.array([self._K])

        else:
            self._X = None
            self._Z = None
            self._N = None
            self._D = None
            self._A = None
            self._A_prior = None
            self._K = None
#            self._K_plus = None
            self._pi = None
            self._Z_sum = None
            self.X_test = None
#            self.mask_train = None
            self.test_idx = None
#            self.error_trace = None
            self._X_test = None
            self._P_prime = None
            self.mask = None
#            self.new_feat = None

#        self.error_trace = self.comm.bcast(self.error_trace)
        self._N = self.comm.bcast(self._N)
        self._D = self.comm.bcast(self._D)
        self._A = self.comm.bcast(self._A)
        self._pi = self.comm.bcast(self._pi)
        self._K = self.comm.bcast(self._K)
#        self._K_plus = self.comm.bcast(self._K_plus)
        self._A_prior = self.comm.bcast(self._A_prior)
        self._K_star = 0
        self._Z_sum = self.comm.bcast(self._Z_sum)
#        self._P_prime = self.comm.bcast(self._P_prime)

        self.data_partition = self.partition_tuple()
        self.part_size_X = tuple([j * self._D for j in self.partition_tuple()])
        self.part_size_Z = tuple([j * self._K for j in self.partition_tuple()])
        self.data_displace_X = self.displacement(self.part_size_X)
        self.data_displace_Z = self.displacement(self.part_size_Z)

        self._X_local = numpy.zeros(self.data_partition[self.rank] * self._D)
        self._Z_local = numpy.zeros(self.data_partition[self.rank] * self._K)
        self.mask_local = numpy.zeros(self.data_partition[self.rank] * self._D)

        self.comm.Scatterv([self._Z, self.part_size_Z, self.data_displace_Z, MPI.DOUBLE], self._Z_local)
        self.comm.Scatterv([self._X, self.part_size_X, self.data_displace_X, MPI.DOUBLE], self._X_local)
        self.comm.Scatterv([self.mask, self.part_size_X, self.data_displace_X, MPI.DOUBLE], self.mask_local)
        self._X_local = self._X_local.reshape((self.data_partition[self.rank], self._D))
        self._Z_local = self._Z_local.reshape((self.data_partition[self.rank], self._K)).astype(int)
        self.mask_local = self.mask_local.reshape((self.data_partition[self.rank], self._D))
        self._X_local_eval = numpy.ma.array(self._X_local, mask=1-self.mask_local)
        self._X_local = numpy.ma.array(self._X_local, mask=self.mask_local)

        assert(self._X_local.shape[0] == self._Z_local.shape[0])
        self._N_p = self._X_local.shape[0]
        assert(self._N_p > 0)

#        if self.rank == 0:
#            error = self._X_local - numpy.dot(self._Z_local, self._A )
#            self.error_trace = numpy.trace(numpy.dot(error.T, error))
#        else:
#            self.error_trace = None

        self._A = self.comm.bcast(self._A)
#        self.error_trace = self.comm.bcast(self.error_trace)
        assert(self._A.shape == (self._K, self._D));

#        local_XZA = self._X_local - numpy.dot(self._Z_local, self._A)
#        local_XZA_square = numpy.dot(local_XZA.T, local_XZA)
#        self.error_local = local_XZA_square

        # initialize the hyper-parameter for sampling _sigma_x
        # a value of None is a gentle way to say "do not sampling _sigma_x"
        assert(sigma_x_hyper_parameter is None or type(sigma_x_hyper_parameter) == tuple);
        self._sigma_x_hyper_parameter = sigma_x_hyper_parameter;

        # initialize the hyper-parameter for sampling _sigma_a
        # a value of None is a gentle way to say "do not sampling _sigma_a"
        assert(sigma_a_hyper_parameter is None or type(sigma_a_hyper_parameter) == tuple);
        self._sigma_a_hyper_parameter = sigma_a_hyper_parameter;
        assert(alpha_hyper_parameter is None or type(alpha_hyper_parameter) == tuple);
        self._alpha_hyper_parameter = alpha_hyper_parameter;
#        self._X = self._X.reshape(self._N, self._D)
#        self._Z = self._Z.reshape(self._N, self._K)
#        self._X = None
#        self._Z = None
#        self.mask = None

        self.local_map_time = []
        self.local_reduce_time = []
        return;

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
            displace = numpy.append([0], numpy.cumsum(partition[:-1])).astype(int)
        else:
            displace = [0]
        return(tuple(displace))

    def initialize_Z(self, p =.5):

        Z = numpy.random.binomial(1, p=p, size=(self._N, self._K))
        Z = Z.astype(numpy.int);

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
        M = numpy.linalg.inv(numpy.dot(Z.transpose(), Z) + (self._sigma_x / self._sigma_a) ** 2 * numpy.eye(K));
        return M

    """
    compute the mean and co-variance, i.e., sufficient statistics, of A
    @param observation_index: a list data type, recorded down the observation indices (column numbers) of A we want to compute
    """
    def init_sufficient_statistics_A(self):
        # compute M = (Z' * Z - (sigma_x^2) / (sigma_a^2) * I)^-1
        M = self.compute_M(self._Z);
        # compute the mean of the matrix A
        mean_A = numpy.dot(M, numpy.dot(self._Z.transpose(), self._X));
        # compute the co-variance of the matrix A
        std_dev_A = numpy.linalg.cholesky(self._sigma_x ** 2 * M).transpose();

        return (mean_A, std_dev_A)
    """
    center the data, i.e., subtract the mean
    """
    def center_data(self,data):
        (N, D) = data.shape;
        data = data - numpy.tile(data.mean(axis=0), (N, 1));
        return data

    def log_likelihood_X(self, X=None, Z=None, A=None):
        if A is None:
            A = self._A;
        if Z is None:
            Z = self._Z_local;
        if X is None:
            X = self._X_local;

#        non_zero_Z = numpy.copy(Z).sum(axis=0).nonzero()[0]
#        Z = Z[:,non_zero_Z]
#        A = A[non_zero_Z,:]
        assert(X.shape[0] == Z.shape[0]);
        (N, D) = X.shape;
        (N, K) = Z.shape;
        assert(A.shape == (K, D));

        log_likelihood = X - numpy.dot(Z, A);

        (row, column) = log_likelihood.shape;
        if row > column:
            log_likelihood = numpy.ma.trace(numpy.ma.dot(log_likelihood.T, log_likelihood));
        else:
            log_likelihood = numpy.ma.trace(numpy.ma.dot(log_likelihood, log_likelihood.T));

        log_likelihood = -0.5 * log_likelihood / numpy.power(self._sigma_x, 2);
        log_likelihood -= N * D * 0.5 * numpy.log(2 * numpy.pi * numpy.power(self._sigma_x, 2));

        return log_likelihood

    def collapsed_log_likelihood_X(self, Z=None, X=None):

        if Z is None:
            Z = self._Z_local;

        if X is None:
            X = self._X_local

#        mask = 1-X.mask

#        if test: # if evaluating test set
#            rows = X.mask.sum(axis=1).nonzero()[0]
##            rows = numpy.unique(mask.nonzero()[0])
#            X = X[rows,:]
#            X_test_mask = 1- X.mask
#            X = numpy.ma.array(numpy.array(X), mask=X_test_mask)
#            Z = Z[rows,:]

        M = self.compute_M(Z)
        log_det_M = numpy.log(numpy.linalg.det(M));

        assert(X.shape[0] == Z.shape[0]);
        (N, D) = X.shape;
        (N, K) = Z.shape;
        assert(M.shape == (K, K));

        log_likelihood = numpy.eye(N) - numpy.dot(numpy.dot(Z, M), Z.T);
        log_likelihood = -0.5 / (self._sigma_x ** 2) * numpy.ma.trace(numpy.ma.dot(numpy.ma.dot(X.T, log_likelihood), X));
        log_likelihood -= D * (N - K) * numpy.log(self._sigma_x) + K * D * numpy.log(self._sigma_a);
        log_likelihood += 0.5 * D * log_det_M;
        log_likelihood -= 0.5 * N * D * numpy.log(2 * numpy.pi);
        return(log_likelihood)


    def uncollapased_Zn(self, object_index):
        new_m = (self._Z_sum - self._Z_local[object_index, :]).astype(numpy.float);
        ucs_prob_z1 = numpy.log(self._pi)
        ucs_prob_z0 = numpy.log(1.0 - self._pi)
        singleton_features = [nk for nk in range(self._K) if self._Z_local[object_index, nk] != 0 and new_m[nk] == 0];
        non_singleton_features = [nk for nk in range(self._K) if nk not in singleton_features]
        order = numpy.random.permutation(self._K);
        for (feature_counter, feature_index) in enumerate(order):
            if feature_index in non_singleton_features:
                self._Z_local[object_index, feature_index] = 0;
                prob_z0 = self.log_likelihood_X(self._X_local[[object_index], :], self._Z_local[[object_index], :]);
                prob_z0 += ucs_prob_z0[feature_index];

                self._Z_local[object_index, feature_index] = 1;
                prob_z1 = self.log_likelihood_X(self._X_local[[object_index], :], self._Z_local[[object_index], :]);
                prob_z1 += ucs_prob_z1[feature_index]

                prob = numpy.array([prob_z0, prob_z1])
                max_p = numpy.max(prob)
                log_sum_exp = max_p + numpy.log(numpy.sum(numpy.exp(prob - max_p)))
                Znk_is_0 = prob_z0 - log_sum_exp
                if numpy.log(numpy.random.random()) < Znk_is_0:
                    self._Z_local[object_index, feature_index] = 0;
                else:
                    self._Z_local[object_index, feature_index] = 1;
#        return singleton_features;


    """
    sample alpha from conjugate posterior
    """
    def sample_alpha(self):
        assert(self._alpha_hyper_parameter != None);
        assert(type(self._alpha_hyper_parameter) == tuple);

        (alpha_hyper_a, alpha_hyper_b) = self._alpha_hyper_parameter;

        posterior_shape = alpha_hyper_a + self._K;
        H_N = numpy.array([range(self._N)]) + 1.0;
        H_N = numpy.sum(1.0 / H_N);
        posterior_scale = 1.0 / (alpha_hyper_b + H_N);

        alpha_new = numpy.random.gamma(posterior_shape, posterior_scale);

        return alpha_new;


    def map_step(self):
        start_time = time.time()
        N_p_order = numpy.random.permutation(self._N_p)
        for (object_counter, object_index) in enumerate(N_p_order):
            self.uncollapased_Zn(object_index)
        map_time = time.time() - start_time
        self.local_map_time.append(map_time)

    def reduce_step(self):
        start_time = time.time()
        local_sum = self._Z_local.sum(axis=0)
        self._Z_sum = self.comm.allreduce(local_sum)
        Z_reduce_local = numpy.copy(self._Z_local)#[:, self._Z_sum.nonzero()[0]]

        local_M_inv = numpy.dot(Z_reduce_local.T, Z_reduce_local) # + (self._sigma_x / self._sigma_a) ** 2 * numpy.eye(self._K);
        M = numpy.linalg.inv(self.comm.allreduce(local_M_inv) + (self._sigma_x / self._sigma_a) ** 2 * numpy.eye(self._K))
        local_MZX = numpy.ma.dot(M, numpy.ma.dot(Z_reduce_local.T, self._X_local))
        local_XZA = self._X_local - numpy.ma.dot(Z_reduce_local, self._A)
        local_XZA_square = numpy.ma.dot(local_XZA.T, local_XZA)

        MZX = self.comm.reduce(local_MZX)
        X_minus_ZA = self.comm.reduce(local_XZA_square)

        if self.rank == 0:
            self._pi = numpy.random.beta(self._Z_sum, 1. + self._N - self._Z_sum)
            self._alpha = self.sample_alpha()

            A_sigma_chol = numpy.linalg.cholesky(self._sigma_x**2 * M)
            self._A = MZX + numpy.dot(A_sigma_chol, numpy.random.normal(size = (self._K, self._D)))
            error_trace = numpy.ma.trace(X_minus_ZA)
            gamma_x_a = self._sigma_x_hyper_parameter[0] + (.5 * self._N * self._D)
            gamma_x_b = 1. / (self._sigma_x_hyper_parameter[1] + (.5 * error_trace))
            self._sigma_x = 1. / numpy.sqrt(numpy.random.gamma(gamma_x_a, gamma_x_b))

            A_minus_prior = self._A - self._A_prior
            A_square = numpy.trace(numpy.dot(A_minus_prior.T, A_minus_prior))
            gamma_a_a = self._sigma_a_hyper_parameter[0] + (.5 * self._K * self._D)
            gamma_a_b = 1. / (self._sigma_a_hyper_parameter[1] + (.5 * A_square))
            self._sigma_a = 1. / numpy.sqrt(numpy.random.gamma(gamma_a_a, gamma_a_b))
#            self._P_prime = numpy.random.choice(xrange(self._P), num_P)
#        else:
#            self.error_trace = None

#        self.error_trace = self.comm.bcast(self.error_trace)
        self._A = self.comm.bcast(self._A)
#        self._P_prime = self.comm.bcast(self._P_prime)
        self._sigma_a = self.comm.bcast(self._sigma_a)
        self._sigma_x = self.comm.bcast(self._sigma_x)
        self._pi = self.comm.bcast(self._pi)
        self._alpha = self.comm.bcast(self._alpha)
        reduce_time = time.time() - start_time
        self.local_reduce_time.append(reduce_time)

#    def train_likelihood(self, X=None):
#        train_array = numpy.zeros(X.mask.sum())
#        train_array -= .5*numpy.log(2*numpy.pi) + (1.-self._K)*numpy.log(self._sigma_x)
#        train_array -= self._K*numpy.log(self._sigma_a)
#        X_train = X[X.mask.sum(axis=1).nonzero()[0],:]
#        N,D = X_train.shape
#        Z_train = self._Z[X.mask.sum(axis=1).nonzero()[0],:]
#        M_train = numpy.dot(Z_train.T, Z_train) + numpy.eye(self._K)* (self._sigma_x / self._sigma_a)**2
#        M_train = numpy.linalg.inv(M_train)
#        IZMZ = numpy.eye(N)-numpy.dot(numpy.dot(Z_train,M_train),Z_train.T)
#        train_array += (-.5/self._sigma_x**2)*((X_train**2)*numpy.tile(numpy.diag(IZMZ),(D,1)).T)[X.mask]
#        train_array -= .5*numpy.log(numpy.tile(Z_train.sum(axis=1), (D,1)).T + (self._sigma_x / self._sigma_a)**2)[X.mask]


    def sample(self, iteration=500, Afile="A_post", logfile="likelihood", A_list = "A_list", totalfeat="total_feat", time_file="timing", plot=False):
        if self.rank==0:
            self.likelihood_iteration = numpy.zeros((iteration, 2))
            self.feature_count = numpy.zeros((iteration, 2))
            start_time = time.time()
        else:
            self.likelihood_iteration = None
            self.feature_count = None
            start_time = None
        self._A_list = []
        for i in xrange(iteration):
            self.map_step()
            self.comm.Barrier()
            self.reduce_step()
            self._A_list.append(numpy.array(self._A))
            gather_Z = self.comm.gather(self._Z_local)
            if self.rank == 0:
                gather_Z = numpy.vstack(gather_Z).reshape(self._N, self._K)[self.test_idx,:]
                total_likelihood = self.collapsed_log_likelihood_X(X=self.X_test, Z = gather_Z)
                elapsed = time.time()-start_time
                self.likelihood_iteration[i,:] = [elapsed, total_likelihood]
                self.feature_count[i,:] = [elapsed, self._K]
#                self.likelihood_iteration = np.vstack((self.likelihood_iteration, (elapsed, total_likelihood)))
                save_dict = {'Z_count':self.feature_count, 'likelihood':self.likelihood_iteration, 'features':self._A_list}
#                    today = datetime.datetime.today().strftime("%Y-%m-%d-%f")

                fname =  "IBP_UGS_P" + str(self._P) + "_"  + ".mat"
                savemat(os.path.abspath(fname),save_dict)

                if self.verbose:
                    print("complete: %i.2%%\tK: %i\tlikelihood: %f" % ((100 * float(i)/float(iteration)), self._K, total_likelihood));
                    print("alpha: %f\tsigma_a: %f\tsigma_x: %f\ttime: %f" % (self._alpha, self._sigma_a, self._sigma_x, elapsed));
                    print(self._Z_sum)
#                    if plot:
#                        for k_i, k_v in enumerate(xrange(self._K)):
#                            plot_index = k_v+ 1
#                            sp = plt.subplot(1,self._K, plot_index)
#                            sp.imshow(self._A[k_v].reshape(numpy.sqrt(self._D), numpy.sqrt(self._D)), interpolation="none", cmap='gist_gray')
#                            sp.axis('off')
#                        plt.show()

        total_reduce = self.comm.gather(self.local_reduce_time)
        total_map = self.comm.gather(self.local_map_time)
        if self.rank == 0:
            self.likelihood_iteration = self.likelihood_iteration[self.likelihood_iteration[:,0].nonzero()[0],:] # take out zeros
            numpy.save(Afile+"_"+str(self._P)+"_"+self.today.strftime("%Y-%m-%d-%f"), numpy.array(self._A))
            numpy.save(logfile+"_"+str(self._P)+"_"+self.today.strftime("%Y-%m-%d-%f"), self.likelihood_iteration)
            numpy.save(A_list+"_"+str(self._P)+"_"+self.today.strftime("%Y-%m-%d-%f"), self._A_list)
            timing = {'reduce':total_reduce , 'total_map':total_map}
            numpy.save(time_file+"_"+str(self._P)+"_"+self.today.strftime("%Y-%m-%d-%f"), timing)
        else:
            timing = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel inference in Beta-Bernoulli Process")
    parser.add_argument('--data',
                        metavar='filename',
                        help='Data for model, must be numpy N*D array')
    parser.add_argument('--logfile',
                        metavar='filename',
                        help='Filename for log likelihood data')
    parser.add_argument('--afile',
                        metavar='filename',
                        help='Filename for feature posterior data')
    parser.add_argument('--alist', metavar='filename',
                        help='Fileame for list of A over sub-iterations')
    parser.add_argument('--verbose',
                        action="store_true", default=False,
                        help='Flag turns on verbose output')
    parser.add_argument('--iters',
                        metavar='int', type=int,
                        help='number of iteraitons, default 1000')
    parser.add_argument('--K',
                        metavar='int', type=int,
                        help='number of features to instantiate')
    parser.add_argument('--newfeat',
                        metavar='filename',
                        help='Filename for features added per global step')
    parser.add_argument('--timing',
                        metavar='filename',
                        help='Filename for timing data')
    parser.add_argument('--pkl',
                        metavar='filename',
                        help='Filename for pickle data')
    parser.add_argument('--initialZ',
                        metavar='filename', default=None,
                        help='Path and filename for initial values for Z, defaults to None')


#    parser.set_defaults(data = os.path.abspath("../research/hybridIBP/data/cambridge.npy"), logfile = os.path.abspath("../research/hybridIBP/output/log_likelihood_parallel"),
#                        afile = os.path.abspath("../research/hybridIBP/output/posterior_A_parallel"), verbose=False, iters=1000, L = 5)
    parser.set_defaults(data = os.path.abspath("../data/cambridge_10k.npy"), logfile = os.path.abspath("../output/log_likelihood_parallel",), initialZ = os.path.abspath("init_Z.npy"),
                        afile = os.path.abspath("../output/posterior_A_parallel"), verbose=True, iters=1000, alist=os.path.abspath("../output/A_list"),
                        newfeat = os.path.abspath("../output/newfeat"), timing= os.path.abspath("../output/timing"), pkl=os.path.abspath("../output/IBP.pkl"), K=5)

    args = parser.parse_args()

    pickle_name = os.path.abspath(args.pkl)
#    X=numpy.load(os.path.abspath(args.data))[:500]
    X=numpy.load(os.path.abspath(args.data))

    X_length = X.shape[0]
    log_name = os.path.abspath(args.logfile)
    A_name = os.path.abspath(args.afile)
    verbose_flag = args.verbose
    a_list = args.alist
    feat_name = os.path.abspath(args.newfeat)
    time_name = os.path.abspath(args.timing)
    if args.initialZ is not None:
        Z = numpy.load(os.path.abspath(args.initialZ))
    else:
        Z = None


#    hybrid = HybridIBPAll(data=X, initial_Z = numpy.random.binomial(1, .5, size=(X_length,2)),verbose=verbose_flag)
    hybrid = BetaBernoulli(data=X,verbose=verbose_flag, K=args.K,initial_Z = Z)
#    hybrid = HybridIBP(data=X, initial_Z = numpy.random.binomial(1, .5, size=(X_length,2)),verbose=True)
    hybrid.sample(iteration=args.iters, Afile= A_name, logfile=log_name, A_list=a_list, totalfeat=feat_name, time_file=time_name)
#    if hybrid.rank==0:
#        joblib.dump(hybrid,pickle_name, compress=9)