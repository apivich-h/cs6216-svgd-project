import numpy as np
import matplotlib.pyplot as plt
import tqdm

import torch
import pyro
import pyro.distributions as dist
from pyro.optim import Adam

from pyro.infer.svgd import SVGD, RBFSteinKernel
from pyro.infer.mcmc import MCMC
from pyro.infer.mcmc.nuts import NUTS
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal

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
    # prior distributions for alpha and w
    alpha = pyro.sample("alpha", dist.Gamma(1., 0.01))
    w = pyro.sample("w", dist.Normal(torch.zeros(2), (1. / alpha) * torch.ones(2)))
    with pyro.plate("data", len(x_data)):
        logits = torch.sum(x_data * w, dim=-1)
        return pyro.sample("obs", dist.Bernoulli(logits=logits), obs=y_data)


if __name__ == '__main__':

    # dataset
    xs, ys = generate_gmm()
    xs_tens = torch.Tensor(xs)
    ys_tens = torch.Tensor(ys)

    # for plotting purposes
    steps_pic = [1, 50, 100, 150]
    x0_sp = np.linspace(-4., 4.)
    fig = plt.figure(figsize=(6 * (len(steps_pic) + 1), 5))
    axs = fig.subplots(nrows=1, ncols=len(steps_pic) + 2, sharex=True, sharey=True)
    axs[0].scatter(xs[ys == 0, 0], xs[ys == 0, 1], s=2)
    axs[0].scatter(xs[ys == 1, 0], xs[ys == 1, 1], s=2)
    axs[0].set_xlim((-4, 4))
    axs[0].set_ylim((-4, 4))
    pic_i = 1

    # SVGD ALGORITHM

    num_particles = 100
    steps_pic = [1, 50, 100, 150]

    pyro.clear_param_store()
    print("SVGD")

    kernel = RBFSteinKernel()
    adam = Adam({"lr": 0.1})
    svgd = SVGD(logistic_reg_model, kernel, adam, num_particles=num_particles, max_plate_nesting=2)

    for i in tqdm.trange(steps_pic[-1] + 1):

        svgd.step(xs_tens, ys_tens)

        if i in steps_pic:
            particles = svgd.get_named_particles()

            ax = axs[pic_i]
            ax.scatter(xs[ys == 0, 0], xs[ys == 0, 1], s=2)
            ax.scatter(xs[ys == 1, 0], xs[ys == 1, 1], s=2)
            for w in particles['w']:
                wi_0, wi_1 = w.detach().numpy()[0]
                ax.plot(x0_sp, -wi_0 * x0_sp / wi_1, color='black', alpha=min(1, 20. / n))

            ax.set_title(f'Iteration {i}')
            pic_i += 1

    # NUTS ALGORITHM
    pyro.clear_param_store()
    print("NUTS")

    n_nuts = 1000
    nuts_kernel = NUTS(logistic_reg_model, adapt_step_size=True)
    mcmc = MCMC(nuts_kernel, num_samples=n_nuts, warmup_steps=n_nuts)
    mcmc.run(xs_tens, ys_tens)

    ax_nuts = axs[pic_i]
    ax_nuts.set_title("NUTS")
    ax_nuts.scatter(xs[ys == 0, 0], xs[ys == 0, 1], s=2)
    ax_nuts.scatter(xs[ys == 1, 0], xs[ys == 1, 1], s=2)
    for w_sample in mcmc.get_samples(n_nuts)['w']:
        wi_0, wi_1 = w_sample.detach().numpy()
        ax_nuts.plot(x0_sp, -wi_0 * x0_sp / wi_1, color='black', alpha=0.005)
    pic_i += 1

    # # VI ALGORITHM
    # pyro.clear_param_store()
    # print("SVI")
    #
    # adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
    # optimizer = Adam(adam_params)
    # svi = SVI(logistic_reg_model, AutoNormal(logistic_reg_model), optimizer, loss=Trace_ELBO())
    # n_steps = steps_pic[-1]
    # # do gradient steps
    # for step in tqdm.trange(n_steps):
    #     svi.step(xs_tens, ys_tens)
    #
    # ax_svi = axs[-1]
    # ax_svi.set_title("SVI")
    # ax_svi.scatter(xs[ys == 0, 0], xs[ys == 0, 1], s=2)
    # ax_svi.scatter(xs[ys == 1, 0], xs[ys == 1, 1], s=2)
    # w_0, w_1 = pyro.param("w0").detach().numpy()
    # ax_svi.plot(x0_sp, -w_0 * x0_sp / w_1, color='black')

    # fig.tight_layout()
    fig.savefig('figs/bayes_logistic')
