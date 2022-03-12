import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
import numpy.matlib as nm

import numpy as np
from scipy.spatial.distance import pdist, squareform
import pdb


class SVGD():
    def __init__(self):
        pass

    def svgd_kernel(self, theta, h=-1):
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

    def update(self, x0, lnprob, n_iter=1000, stepsize=1e-3, bandwidth=-1, alpha=0.9, debug=False):
        # Check input
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')

        theta = np.copy(x0)
        theta_hist = []

        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        for iter in range(n_iter):
            if debug and (iter + 1) % 1000 == 0:
                print('iter ' + str(iter + 1))

            lnpgrad = lnprob(theta)
            # calculating the kernel matrix
            kxy, dxkxy = self.svgd_kernel(theta, h=-1)
            grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]

            # adagrad
            if iter == 0:
                historical_grad = historical_grad + grad_theta ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
            adj_grad = np.divide(grad_theta, fudge_factor + np.sqrt(historical_grad))
            theta = theta + stepsize * adj_grad

            if (iter+1) % 1000 == 0:
                theta_hist.append(theta)
        return theta, theta_hist


'''
Implementation of Stochastic Gradient Langevin Dynamics from https://icml.cc/2011/papers/398_icmlpaper.pdf.
Inspired by implementaion of SGLD from https://github.com/wiseodd/MCMC/blob/master/algo/sgld.py.
'''
class SLGD():
    def __init__(self):
        pass

    def update(self, x0, lnprob, n_iter, N, n, bandwidth=-1, alpha=0.9, a=1.0, debug=False):
        # Check input
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')

        theta = np.copy(x0)
        theta_hist = []

        for iter in range(n_iter):
            if debug and (iter + 1) % 1000 == 0:
                print('iter ' + str(iter + 1))

            lnpgrad = lnprob(theta)
            step_size = 1.0/((iter+2)**0.55)
            grad_logprior = theta
            grad = grad_logprior + N/n * lnpgrad
            adj_grad = step_size/2 * grad + np.random.normal(0, np.sqrt(step_size))
            theta = theta + step_size * adj_grad

            if (iter+1) % 1000 == 0:
                theta_hist.append(theta)
        return theta, theta_hist


class BayesianLR:
    def __init__(self, X, Y, batchsize=50, a0=1, b0=0.01):
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
        # llh = np.mean(np.log(prob))
        return acc


if __name__ == '__main__':
    data = scipy.io.loadmat('original-code/Stein-Variational-Gradient-Descent/data/covertype.mat')

    X_input = data['covtype'][:, 1:]
    y_input = data['covtype'][:, 0]
    y_input[y_input == 2] = -1

    N = X_input.shape[0]
    X_input = np.hstack([X_input, np.ones([N, 1])])
    d = X_input.shape[1]
    D = d + 1

    method = "SGLD"

    accuracy = []
    # split the dataset into training and testing
    for trial in range(1):
        acc = []
        print("Trial " + str(trial+1))
        X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.2)

        a0, b0 = 1, 0.01  # hyper-parameters
        model = BayesianLR(X_train, y_train, 50, a0, b0)  # batchsize = 50

        # initialization
        M = 100  # number of particles
        theta0 = np.zeros([M, D]);
        alpha0 = np.random.gamma(a0, b0, M);
        for i in range(M):
            theta0[i, :] = np.hstack([np.random.normal(0, np.sqrt(1 / alpha0[i]), d), np.log(alpha0[i])])
        if method == "SVGD":
            theta, theta_hist = SVGD().update(x0=theta0, lnprob=model.dlnprob, bandwidth=-1, n_iter=18000, stepsize=0.05, alpha=0.9,
                              debug=True)
        elif method == "SGLD":
            theta, theta_hist = SLGD().update(x0=theta0, lnprob=model.dlnprob, N=100, n=50, n_iter=18000, debug=True)
        # theta_hist.append(theta)
        for th in theta_hist:
            acc.append(model.evaluation(th, X_test, y_test))
        accuracy.append(acc)
    accuracy = np.array(accuracy)
    # accuracy = accuracy.mean(axis=0)
    np.savetxt("results/sgld_covtype_2epochs_100particles.csv", accuracy, delimiter=",")
