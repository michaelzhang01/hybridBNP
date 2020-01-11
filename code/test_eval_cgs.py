"""
@author: Ke Zhai (zhaike@cs.umd.edu)

Implements collapsed Gibbs sampling for the linear-Gaussian infinite latent feature model (IBP).
"""
import time
import argparse
import numpy, scipy;
import random;
import os
from gs import GibbsSampling;
import datetime
from scipy.io import savemat
#import matplotlib.pyplot as P
#from util.scaled_image import scaledimage

# We will be taking log(0) = -Inf, so turn off this warning
numpy.seterr(divide='ignore')



class CollapsedGibbsSampling(GibbsSampling):
    import scipy.stats;

    """
    @param data: a NxD NumPy data matrix
    @param alpha: IBP hyper parameter
    @param sigma_x: standard derivation of the noise on data, often referred as sigma_n as well
    @param sigma_a: standard derivation of the feature, often referred as sigma_f as well
    @param initializ_Z: seeded Z matrix
    """
    def _initialize(self, data, alpha=1.0, sigma_a=1.0, sigma_x=1.0,
                    initial_Z=None, A_prior=None, train=False, verbose=False):
        self._alpha = alpha;
        self._sigma_x = sigma_x;
        self._sigma_a = sigma_a;
        self.train = train
        self.verbose = verbose
        self.today = datetime.datetime.today()
        # Data matrix
        #self._X = self.center_data(data);
        self._X = self.center_data(data);
        (self._N, self._D) = self._X.shape;

        if self.train:
            if self._D % 2 == 0:
                even_mask = tuple([0,1]*(self._D//2))
                odd_mask = tuple([1,0]*(self._D//2))
            else:
                even_mask = [0,1]*(self._D//2 + 1)
                even_mask = tuple(even_mask[:self._D])
                odd_mask = [1,0]*(self._D//2 + 1 )
                odd_mask = tuple(odd_mask[:self._D])
            idx_mask = numpy.array([even_mask,odd_mask] * (int(self._N * .1)/ int(2)))
            idx_N = idx_mask.shape[0]
            zeros_mask = numpy.zeros((self._N - idx_N, self._D))
            self.mask = numpy.vstack((idx_mask, zeros_mask)).astype(float)
            assert(self.mask.shape == self._X.shape)
            self.test_idx = self.mask.sum(axis=1).nonzero()[0]
            self.X_test = numpy.ma.array(self._X[self.test_idx,:], mask=1-idx_mask)
            self._X = numpy.ma.array(self._X, mask=self.mask)

        else:
            self.X_test = None
            self.mask = None
            self.test_idx = None

        if(initial_Z is None):
            # initialize Z from IBP(alpha)
            self._Z = self.initialize_Z();
        else:
            self._Z = initial_Z;

        assert(self._Z.shape[0] == self._N);

        # make sure Z matrix is a binary matrix
#        assert(self._Z.dtype == numpy.int);
#        assert(self._Z.max() == 1 and self._Z.min() == 0);

        # record down the number of features
        self._K = self._Z.shape[1];

        if A_prior is None:
            self._A_prior = numpy.zeros((1, self._D));
        else:
            self._A_prior = A_prior;
        assert(self._A_prior.shape == (1, self._D));

        self._A = self.map_estimate_A();
        assert(self._A.shape == (self._K, self._D));

        # compute matrix M
        self._M = self.compute_M();
        self._log_det_M = numpy.log(numpy.linalg.det(self._M));

        assert(numpy.abs(numpy.log(numpy.linalg.det(self._M)) - self._log_det_M) < 0.000000001)

    """
    sample the corpus to train the parameters
    """
    def sample(self, iteration):
        assert(self._Z.shape == (self._N, self._K));
        assert(self._X.shape == (self._N, self._D));
        self.likelihood_iteration = numpy.zeros((iteration, 2))
        self.num_features = numpy.zeros((iteration, 2))
        start_time = time.time()
        #sample the total data
        for iter in xrange(iteration):
            # sample every object
            order = numpy.random.permutation(self._N);
            for (object_counter, object_index) in enumerate(order):
                ziM = numpy.ma.dot(self._Z[[object_index], :], self._M);
                ziMzi = numpy.ma.dot(ziM, self._Z[[object_index], :].transpose());
                M_i = self._M - numpy.ma.dot(ziM.transpose(), ziM) / (ziMzi - 1);
                log_det_M_i = self._log_det_M - numpy.log(1 - ziMzi);

                # sample Z_n
                singleton_features = self.sample_Zn(object_index, M_i, log_det_M_i);
                self.metropolis_hastings_K_new(object_index, singleton_features, M_i, log_det_M_i);

            mean_a, chol_a = self.sufficient_statistics_A();
            self._A = mean_a + numpy.ma.dot(chol_a, numpy.random.normal(size=(self._K, self._D)))
            self._alpha = self.sample_alpha();
            self._sigma_x = self.sample_sigma_x(self._sigma_x_hyper_parameter);
            self._sigma_a = self.sample_sigma_a(self._sigma_a_hyper_parameter);


            if self.train:
                Z_train = self._Z[self.test_idx,:]
                M_train = self.compute_M(Z_train)
                sgn, log_det_M_train = numpy.linalg.slogdet(M_train)
                total_likelihood = self.log_likelihood_X(X=self.X_test, Z=Z_train, M = M_train, log_det_M=log_det_M_train)
            else:
                total_likelihood = self.log_likelihood_model()

            if self.verbose:
                print("iteration: %i\tK: %i\tlikelihood: %f" % (iter, self._K, total_likelihood));
                print("alpha: %f\tsigma_a: %f\tsigma_x: %f" % (self._alpha, self._sigma_a, self._sigma_x));
            self.num_features[iter,:] = [iter, self._K]
            self.likelihood_iteration[iter,:] = [time.time()-start_time, total_likelihood]
            save_dict = {'Z_count':self.num_features, 'likelihood':self.likelihood_iteration, 'features':self._A}
#                    today = datetime.datetime.today().strftime("%Y-%m-%d-%f")

            fname =  "IBP_CGS.mat"
            savemat(os.path.abspath(fname),save_dict)


    """
    @param object_index: an int data type, indicates the object index (row index) of Z we want to sample
    """
    def sample_Zn(self, object_index, M_i, log_det_M_i):
        assert(type(object_index) == int or type(object_index) == numpy.int32 or type(object_index) == numpy.int64);

        # calculate initial feature possess counts
        m = self._Z.sum(axis=0);

        # remove this data point from m vector
        new_m = (m - self._Z[object_index, :]).astype(numpy.float);

        # compute the log probability of p(Znk=0 | Z_nk) and p(Znk=1 | Z_nk)
        log_prob_z1 = numpy.log(new_m / self._N);
        log_prob_z0 = numpy.log(1 - new_m / self._N);

        # find all singleton features possessed by current object
        singleton_features = [nk for nk in range(self._K) if self._Z[object_index, nk] != 0 and new_m[nk] == 0];
        non_singleton_features = [nk for nk in range(self._K) if nk not in singleton_features]
        order = numpy.random.permutation(self._K);

        for (feature_counter, feature_index) in enumerate(order):
            if feature_index in non_singleton_features:
                old_Znk = self._Z[object_index, feature_index];

                # compute the log likelihood when Znk=1
                self._Z[object_index, feature_index] = 1;
                if old_Znk == 0:
                    ziMi = numpy.dot(self._Z[[object_index], :], M_i);
                    ziMizi = numpy.dot(ziMi, self._Z[[object_index], :].transpose());
                    M_tmp_1 = M_i - numpy.dot(ziMi.transpose(), ziMi) / (ziMizi + 1);
                    log_det_M_tmp_1 = log_det_M_i - numpy.log(ziMizi + 1);

                    #assert(numpy.abs(log_det_M_tmp_1 - numpy.log(numpy.linalg.det(M_tmp_1))) < 0.000000001);
                else:
                    M_tmp_1 = self._M;
                    log_det_M_tmp_1 = self._log_det_M;

                prob_z1 = self.log_likelihood_X(M_tmp_1, log_det_M_tmp_1);
                # add in prior
                prob_z1 += log_prob_z1[feature_index];

                # compute the log likelihood when Znk=0
                self._Z[object_index, feature_index] = 0;
                if old_Znk == 1:
                    ziMi = numpy.dot(self._Z[[object_index], :], M_i);
                    ziMizi = numpy.dot(ziMi, self._Z[[object_index], :].transpose());
                    M_tmp_0 = M_i - numpy.dot(ziMi.transpose(), ziMi) / (ziMizi + 1);
                    log_det_M_tmp_0 = log_det_M_i - numpy.log(ziMizi + 1);
                else:
                    M_tmp_0 = self._M;
                    log_det_M_tmp_0 = self._log_det_M;

                prob_z0 = self.log_likelihood_X(M_tmp_0, log_det_M_tmp_0);
                # add in prior
                prob_z0 += log_prob_z0[feature_index];

                #print "propose znk to 0", numpy.exp(prob_z1-prob_z0);
                Znk_is_0 = 1 / (1 + numpy.exp(prob_z1 - prob_z0));
                #print "znk is 0 with prob", Znk_is_0
                if random.random() < Znk_is_0:
                    self._Z[object_index, feature_index] = 0;
                    self._M = M_tmp_0;
                    self._log_det_M = log_det_M_tmp_0;
                else:
                    self._Z[object_index, feature_index] = 1;
                    self._M = M_tmp_1;
                    self._log_det_M = log_det_M_tmp_1;

        return singleton_features;

    """
    sample K_new using metropolis hastings algorithm
    """
    def metropolis_hastings_K_new(self, object_index, singleton_features, M_i, log_det_M_i):
        # sample K_new from the metropolis hastings proposal distribution, i.e., a poisson distribution with mean \frac{\alpha}{N}
        K_temp = scipy.stats.poisson.rvs(self._alpha / self._N);

        if K_temp <= 0 and len(singleton_features) <= 0:
            return False;

        # compute the probability of using old features
        prob_old = self.log_likelihood_X();

        # construct Z_new
        #Z_new = self._Z[:, [k for k in range(self._K) if k not in singleton_features]];
        Z_new = numpy.hstack((self._Z, numpy.zeros((self._N, K_temp))));
        Z_new[[object_index], [xrange(-K_temp, 0)]] = 1;
        Z_new[[object_index], singleton_features] = 0;

        # construct M_new
        M_i_new = numpy.vstack((numpy.hstack((M_i, numpy.zeros((self._K, K_temp)))), numpy.hstack((numpy.zeros((K_temp, self._K)), (self._sigma_a / self._sigma_x) ** 2 * numpy.eye(K_temp)))));
        log_det_M_i_new = log_det_M_i + 2 * K_temp * numpy.log(self._sigma_a / self._sigma_x);
        ziMi = numpy.dot(Z_new[[object_index], :], M_i_new);
        ziMizi = numpy.dot(ziMi, Z_new[[object_index], :].transpose());
        M_new = M_i_new - numpy.dot(ziMi.transpose(), ziMi) / (ziMizi + 1);
        log_det_M_new = log_det_M_i_new - numpy.log(ziMizi + 1);
        K_new = self._K + K_temp;
        #assert(numpy.abs(log_det_M_new - numpy.log(numpy.linalg.det(M_new))) < 0.000000001);

        # compute the probability of using new features
        prob_new = self.log_likelihood_X(M_new, log_det_M_new, Z_new);

        # compute the probability of generating new features
        accept_new = 1 / (1 + numpy.exp(prob_old - prob_new));

        # if we accept the proposal, we will replace old A and Z matrices
        if random.random() < accept_new:
            self._Z = Z_new;
            self._K = K_new;
            self.regularize_matrices();
            return True;

        return False;

    """
    remove the empty column in matrix Z and the corresponding feature in A
    """
    def regularize_matrices(self):
        assert(self._Z.shape == (self._N, self._K));
        Z_sum = numpy.sum(self._Z, axis=0);
        assert(len(Z_sum) == self._K);
        indices = numpy.nonzero(Z_sum == 0);

        self._Z = self._Z[:, [k for k in range(self._K) if k not in indices]];
        self._K = self._Z.shape[1];
        assert(self._Z.shape == (self._N, self._K));

        # compute matrix M
        self._M = self.compute_M();
        self._log_det_M = numpy.log(numpy.linalg.det(self._M));

#    def sufficient_statistics_A(self):
#        # compute M = (Z' * Z - (sigma_x^2) / (sigma_a^2) * I)^-1
#        M = self.compute_M();
#        # compute the mean of the matrix A
#        mean_A = numpy.dot(M, numpy.dot(self._Z.transpose(), self._X));
#        # compute the co-variance of the matrix A
#        std_dev_A = numpy.linalg.cholesky(self._sigma_x ** 2 * M).transpose();
#
#        return (mean_A, std_dev_A)

    """
    compute the log-likelihood of the data X
    @param X: a 2-D numpy array
    @param Z: a 2-D numpy boolean array
    @param A: a 2-D numpy array, integrate A out if it is set to None
    """
    def log_likelihood_X(self, M=None, log_det_M=None, Z=None, X=None):
        if M is None:
            M = self._M;
            if log_det_M is None:
                log_det_M = numpy.log(numpy.linalg.det(M));
            else:
                log_det_M = self._log_det_M;

        if Z is None:
            Z = self._Z;

        if X is None:
            X = self._X


        assert(X.shape[0] == Z.shape[0]);
        (N, D) = X.shape;
        (N, K) = Z.shape;
        assert(M.shape == (K, K));

        log_likelihood = numpy.eye(N) - numpy.ma.dot(numpy.ma.dot(Z, M), Z.T);
        log_likelihood = -0.5 / (self._sigma_x ** 2) * numpy.ma.trace(numpy.ma.dot(numpy.ma.dot(X.T, log_likelihood), X));
        log_likelihood -= D * (N - K) * numpy.log(self._sigma_x) + K * D * numpy.log(self._sigma_a);
        log_likelihood += 0.5 * D * log_det_M;
        log_likelihood -= 0.5 * N * D * numpy.log(2 * numpy.pi);

        return log_likelihood

    """
    compute the log-likelihood of the model
    """
    def log_likelihood_model(self, X=None):
        #print self.log_likelihood_X(self._X, self._Z, self._A_mean), self.log_likelihood_A(), self.log_likelihood_Z();
        return self.log_likelihood_X(X=X) + self.log_likelihood_Z();

    """
    sample noise variances, i.e., sigma_x
    """
    def sample_sigma_x(self, sigma_x_hyper_parameter):
        return self.sample_sigma(self._sigma_x_hyper_parameter, self._X - numpy.ma.dot(self._Z, self._A));

    """
    sample feature variance, i.e., sigma_a
    """
    def sample_sigma_a(self, sigma_a_hyper_parameter):
        return self.sample_sigma(self._sigma_a_hyper_parameter, self._A);

"""
run IBP on the synthetic 'cambridge bars' dataset, used in the original paper.
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collapsed inference in Indian Buffet Process.")
    parser.add_argument('--data',
                        metavar='filename',
                        help='Data for model, must be numpy N*D array')
    parser.add_argument('--logfile',
                        metavar='filename',
                        help='Filename for log likelihood data')
    parser.add_argument('--test_off',
                        action="store_false", default=True,
                        help='Turn off test set evaluation')
    parser.add_argument('--verbose',
                        action="store_true", default=False,
                        help='Flag turns on verbose output')
    parser.add_argument('--iters',
                        metavar='int', type=int,
                        help='number of iteraitons, default 1000')
    parser.add_argument('--features',
                        metavar='filename',
                        help='Filename for features')
    parser.add_argument('--newfeat',
                        metavar='filename',
                        help='Filename for new feature count')


#    parser.set_defaults(data = os.path.abspath("../research/hybridIBP/data/cambridge.npy"), logfile = os.path.abspath("../research/hybridIBP/output/log_likelihood_collapsed"),
#                        test_off=True, verbose=False, iters=1000)
    parser.set_defaults(data = os.path.abspath("../data/cambridge_10k.npy"), logfile = os.path.abspath("../output/collapsed_likelihood"),
                        test=False, verbose=True, iters=1500, features=os.path.abspath("../output/collapsed_features"),
                        newfeat=os.path.abspath("../output/collapsed_newfeats"))


    args = parser.parse_args()
    today = datetime.datetime.today()
#    block = scipy.io.loadmat("../data/block_image_set")
#    true_weights = block['trueWeights']
    X = numpy.load(args.data)
    X_length = X.shape[0]
    log_name = args.logfile +"_"+today.strftime("%Y-%m-%d-%f")
    test_mode = args.test_off
    verbose_flag = args.verbose
    feat_name = args.features + "_"+today.strftime("%Y-%m-%d-%f")
    newfeat_name = args.newfeat + "_"+today.strftime("%Y-%m-%d-%f")

    alpha_hyper_parameter = (1., 1.);
    sigma_x_hyper_parameter = (1., 1.);
    sigma_a_hyper_parameter = (1., 1.);

    try:
        initial_Z = numpy.load("init_Z.npy")
    except:
        initial_Z = numpy.random.binomial(1, .5, (X_length,2))

    ibp = CollapsedGibbsSampling(alpha_hyper_parameter, sigma_x_hyper_parameter, sigma_a_hyper_parameter, True);
    ibp._initialize(data=X, initial_Z = initial_Z, train=test_mode, verbose=verbose_flag)

    ibp.sample(args.iters);
    numpy.save(log_name,ibp.likelihood_iteration)
    numpy.save(feat_name, numpy.array(ibp._A))
    numpy.save(newfeat_name, ibp.num_features)

