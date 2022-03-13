import numpy as np
import numpy.matlib as nm
import time
from scipy.spatial.distance import pdist, squareform
import tqdm


class SVGD:

    def __init__(self):
        pass

    @staticmethod
    def svgd_kernel(theta, h=-1):
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist) ** 2
        if h < 0:  # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(theta.shape[0] + 1))

        # compute the rbf kernel
        Kxy = np.exp(-pairwise_dists / h ** 2 / 2)

        dxkxy = -np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:, i] + np.multiply(theta[:, i], sumkxy)
        dxkxy = dxkxy / (h ** 2)
        return (Kxy, dxkxy)

    @staticmethod
    def update(x0, lnprob, n_iter=1000, stepsize=1e-3, bandwidth=-1, alpha=0.9, debug=False):
        # Check input
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')

        theta = np.copy(x0)

        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        for iter in tqdm.trange(n_iter):
            if debug and (iter + 1) % 1000 == 0:
                print('iter ' + str(iter + 1))

            lnpgrad = lnprob(theta)
            # calculating the kernel matrix
            kxy, dxkxy = SVGD.svgd_kernel(theta, h=-1)
            grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]

            # adagrad
            if iter == 0:
                historical_grad = historical_grad + grad_theta ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
            adj_grad = np.divide(grad_theta, fudge_factor + np.sqrt(historical_grad))
            theta = theta + stepsize * adj_grad

        return theta


'''
    Example of Bayesian Logistic Regression (the same setting as Gershman et al. 2012):
    The observed data D = {X, y} consist of N binary class labels, 
    y_t \in {-1,+1}, and d covariates for each datapoint, X_t \in R^d.
    The hidden variables \theta = {w, \alpha} consist of d regression coefficients w_k \in R,
    and a precision parameter \alpha \in R_+. We assume the following model:
        p(\alpha) = Gamma(\alpha; a, b)
        p(w_k | a) = N(w_k; 0, \alpha^-1)
        p(y_t = 1| x_t, w) = 1 / (1+exp(-w^T x_t))
'''


class BayesianLR:
    def __init__(self, X, Y, batchsize=100, a0=1, b0=0.01):
        self.X, self.Y = X, Y
        # TODO. Y in \in{+1, -1}
        self.batchsize = min(batchsize, X.shape[0])
        self.a0, self.b0 = a0, b0

        self.N = X.shape[0]
        self.permutation = np.random.permutation(self.N)
        self.iter = 0

    def dlnprob(self, theta):

        if self.batchsize > 0:
            batch = [i % self.N for i in range(self.iter * self.batchsize, (self.iter + 1) * self.batchsize)]
            ridx = self.permutation[batch]
            self.iter += 1
        else:
            ridx = np.random.permutation(self.X.shape[0])

        Xs = self.X[ridx, :]
        Ys = self.Y[ridx]

        w = theta[:, :-1]  # logistic weights
        alpha = np.exp(theta[:, -1])  # the last column is logalpha
        d = w.shape[1]

        wt = np.multiply((alpha / 2), np.sum(w ** 2, axis=1))

        coff = np.matmul(Xs, w.T)
        y_hat = 1.0 / (1.0 + np.exp(-1 * coff))

        dw_data = np.matmul(((nm.repmat(np.vstack(Ys), 1, theta.shape[0]) + 1) / 2.0 - y_hat).T, Xs)  # Y \in {-1,1}
        dw_prior = -np.multiply(nm.repmat(np.vstack(alpha), 1, d), w)
        dw = dw_data * 1.0 * self.X.shape[0] / Xs.shape[0] + dw_prior  # re-scale

        dalpha = d / 2.0 - wt + (self.a0 - 1) - self.b0 * alpha + 1  # the last term is the jacobian term

        return np.hstack([dw, np.vstack(dalpha)])  # % first order derivative

    def evaluation(self, theta, X_test, y_test):
        theta = theta[:, :-1]
        M, n_test = theta.shape[0], len(y_test)

        prob = np.zeros([n_test, M])
        for t in range(M):
            coff = np.multiply(y_test, np.sum(-1 * np.multiply(nm.repmat(theta[t, :], n_test, 1), X_test), axis=1))
            prob[:, t] = np.divide(np.ones(n_test), (1 + np.exp(coff)))

        prob = np.mean(prob, axis=1)
        acc = np.mean(prob > 0.5)
        llh = np.mean(np.log(prob))
        return [acc, llh]


def train_original_svgd(xs_train, ys_train, xs_test, ys_test, num_particles=100,
                        num_steps=6000, lr=0.05, batch_size=100):
    ys_train_copy = np.copy(ys_train)
    ys_train_copy[ys_train == 0] = -1

    ys_test_copy = np.copy(ys_test)
    ys_test_copy[ys_test == 0] = -1

    a0, b0 = 1, 0.01  # hyper-parameters
    model = BayesianLR(xs_train, ys_train_copy, batchsize=batch_size, a0=a0, b0=b0)  # batchsize = 100

    # initialization
    D = xs_train.shape[1] + 1
    theta0 = np.zeros([num_particles, D])
    alpha0 = np.random.gamma(a0, b0, num_particles)
    for i in range(num_particles):
        theta0[i, :] = np.hstack([np.random.normal(0, np.sqrt(1 / alpha0[i]), xs_train.shape[1]), np.log(alpha0[i])])

    t = time.time()
    theta = SVGD.update(x0=theta0, lnprob=model.dlnprob, bandwidth=-1, n_iter=num_steps, stepsize=lr, alpha=0.9,
                        debug=False)
    t = time.time() - t

    acc, llh = model.evaluation(theta, xs_test, ys_test_copy)
    return acc, llh, t
