import numpy as np
import matplotlib.pyplot as plt
import tqdm

import torch
import pyro
import pyro.distributions as dist
from pyro.optim import Adam

from pyro.infer.svgd import SVGD, RBFSteinKernel

n = 500


def generate_gmm(seed=None):
    np.random.seed(seed)

    xs = np.empty((n, 2))
    ys = np.empty((n,))
    for i in range(500):
        y = np.random.binomial(1, 1 / 3)
        if y == 0:
            x = np.random.multivariate_normal([1., 1.], [[1., 0.2], [0.2, 0.4]])
        else:
            x = np.random.multivariate_normal([-1., -1.], [[0.7, 0.3], [0.3, 1.2]])
        xs[i] = x
        ys[i] = y

    return xs, ys


def logistic_reg_model(x_data, y_data=None):
    # TODO: fix alpha into the equation properly
    alpha = pyro.sample("alpha", dist.Gamma(1., 0.01))
    cov = torch.eye(x_data.shape[1])
    # print(cov.shape)
    w = pyro.sample("w", dist.MultivariateNormal(torch.ones(x_data.shape[1]), cov))
    # print(w.shape)
    # raise Exception
    with pyro.plate("data", len(x_data)):
        logits = torch.sum(x_data * w, dim=-1)
        return pyro.sample("obs", dist.Bernoulli(logits=logits), obs=y_data)


if __name__ == '__main__':

    num_particles = 100
    steps_pic = [1, 25, 50, 75, 100, 150]

    pyro.clear_param_store()

    xs, ys = generate_gmm()
    xs_tens = torch.Tensor(xs)
    ys_tens = torch.Tensor(ys)

    kernel = RBFSteinKernel()
    adam = Adam({"lr": 0.1})
    svgd = SVGD(logistic_reg_model, kernel, adam, num_particles=num_particles, max_plate_nesting=1)

    fig = plt.figure(figsize=(5 * (len(steps_pic) + 1), 5))
    axs = fig.subplots(nrows=1, ncols=len(steps_pic) + 1, sharex=True, sharey=True)
    pic_i = 1

    axs[0].scatter(xs[ys == 0, 0], xs[ys == 0, 1], s=2)
    axs[0].scatter(xs[ys == 1, 0], xs[ys == 1, 1], s=2)
    axs[0].set_xlim((-3, 4))
    axs[0].set_ylim((-4, 4))

    for i in tqdm.trange(steps_pic[-1] + 1):

        svgd.step(xs_tens, ys_tens)

        if i in steps_pic:
            particles = svgd.get_named_particles()

            ax = axs[pic_i]
            ax.scatter(xs[ys == 0, 0], xs[ys == 0, 1], s=2)
            ax.scatter(xs[ys == 1, 0], xs[ys == 1, 1], s=2)
            x0_sp = np.linspace(-3., 4.)
            for w in particles['w']:
                wi_0, wi_1 = w.detach().numpy()[0]
                ax.plot(x0_sp, -wi_0 * x0_sp / wi_1, color='black', alpha=min(1, 20. / n))

            ax.set_title(f'Iteration {i}')
            pic_i += 1

    fig.savefig('figs/bayes_logistic_svgd')
