import numpy as np
import matplotlib.pyplot as plt

from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.optim import Adam
from numpyro.infer import Predictive, Trace_ELBO, init_to_uniform, init_to_feasible, init_to_value
from numpyro.infer.autoguide import AutoNormal

from numpyro.contrib.einstein.steinvi import SteinVI
from numpyro.contrib.einstein.kernels import RBFKernel
from numpyro.infer.mcmc import MCMC
from numpyro.infer.hmc import NUTS, HMC
from numpyro.infer.svi import SVI

n = 500


def generate_gmm(dims=2, count=500, seed=None):
    np.random.seed(seed)

    centre = np.sqrt(1 / dims)
    xs = np.empty((count, dims))
    ys = np.empty((count,))
    for i in range(count):
        y = np.random.binomial(1, 1 / 3)
        if y == 0:
            x = np.random.multivariate_normal(centre * np.ones(dims), np.identity(dims))
        else:
            x = np.random.multivariate_normal(- centre * np.ones(dims), np.identity(dims))
        xs[i] = x
        ys[i] = y

    return xs, ys


def logistic_reg_model_generator(dims=2, bias=False):

    def logistic_reg_model(x_data, y_data=None):
        # prior distributions for alpha and w
        alpha = numpyro.sample("alpha", dist.Gamma(1., 0.01))
        with numpyro.plate("wgt", dims, dim=-1):
            w = numpyro.sample("w", dist.Normal(np.zeros(dims), (1. / alpha) * np.ones(dims)))
        if bias:
            b = numpyro.sample("b", dist.Normal(0., 1.))
            with numpyro.plate("data_covertype", len(x_data), dim=-1):
                logits = x_data @ w + b
                numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y_data)
        else:
            with numpyro.plate("data_covertype", len(x_data), dim=-1):
                logits = x_data @ w
                numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y_data)

    return logistic_reg_model


def metrics(ys_test, ys_pred):
    acc = (ys_test * ys_pred + (1 - ys_test) * (1 - ys_pred)).mean()
    abs_acc = (ys_test == (ys_pred > 0.5).astype(np.int32)).astype(np.int32).mean()
    logp = (ys_test * np.log(ys_pred) + (1 - ys_test) * np.log(1 - ys_pred))
    logp = logp[~np.isnan(logp)].mean()
    return acc, abs_acc, logp


if __name__ == '__main__':

    # dataset
    xs, ys = generate_gmm()
    model = logistic_reg_model_generator()

    # for plotting purposes
    # steps_pic = [1, 20]
    steps_pic = [1, 20, 50, 100, 150, 300]
    x0_sp = np.linspace(-4., 4., num=150)
    fig = plt.figure(figsize=(6 * (len(steps_pic) + 1), 5))
    axs = fig.subplots(nrows=1, ncols=len(steps_pic) + 2, sharex=True, sharey=True)
    axs[0].scatter(xs[ys == 0, 0], xs[ys == 0, 1], s=2)
    axs[0].scatter(xs[ys == 1, 0], xs[ys == 1, 1], s=2)
    axs[0].set_xlim((-4, 4))
    axs[0].set_ylim((-4, 4))
    pic_i = 1

    x1_plot, x2_plot = np.meshgrid(x0_sp, x0_sp)

    # SVGD ALGORITHM
    num_particles = 100

    print("SVGD")
    steps_elapsed = 0
    inf_key, pred_key, data_key = random.split(random.PRNGKey(42), 3)
    rng_key, inf_key = random.split(inf_key)

    svgd = SteinVI(
        model=model,
        guide=AutoNormal(model, init_scale=1.),
        kernel_fn=RBFKernel(),
        loss=Trace_ELBO(),
        optim=Adam(0.1),
        num_particles=num_particles
    )

    for step in steps_pic:

        result = svgd.run(rng_key, step, xs, ys)

        pred = Predictive(
            model,
            return_sites=['w'],
            guide=svgd.guide,
            params=svgd.get_params(result.state),
            num_samples=num_particles,
            batch_ndims=1,  # stein particle dimension
        )

        ax = axs[pic_i]
        ax.scatter(xs[ys == 0, 0], xs[ys == 0, 1], s=2)
        ax.scatter(xs[ys == 1, 0], xs[ys == 1, 1], s=2)

        grid = np.zeros_like(x1_plot)
        ws = pred(pred_key, xs)['w']
        for w in ws[0]:
            wi_1, wi_2 = w
            grid += (wi_1 * x1_plot + wi_2 * x2_plot > 0)
            # ax.plot(x0_sp, -wi_0 * x0_sp / wi_1, color='black', alpha=min(1, 20. / count))
        cm = plt.cm.get_cmap('viridis')
        ax.pcolormesh(x1_plot, x2_plot, grid, cmap=cm, alpha=0.3)

        ax.set_title(f'Iteration {step}')
        pic_i += 1

        # pred = Predictive(
        #     model,
        #     guide=svgd.guide,
        #     params=svgd.get_params(result.state),
        #     num_samples=num_particles,
        #     batch_ndims=1,  # stein particle dimension
        # )
        # ys_pred = pred(pred_key, xs)['obs'][0].mean(axis=0)
        # ys_actual = ys.flatten()
        # acc = ((ys_pred > 0.5) == ys_actual).astype(np.int32).mean()
        # logp = (ys_actual * np.log(ys_pred) + (1 - ys_actual) * np.log(1 - ys_pred)).mean()
        # print(acc, logp)

    # NUTS ALGORITHM
    print("NUTS")

    n_nuts = 1000
    inf_key, pred_key, data_key = random.split(random.PRNGKey(42), 3)
    rng_key, inf_key = random.split(inf_key)

    nuts_kernel = NUTS(model, adapt_step_size=True)
    mcmc = MCMC(nuts_kernel, num_samples=n_nuts, num_warmup=n_nuts)
    mcmc.run(rng_key, xs, ys)

    ax_nuts = axs[pic_i]
    ax_nuts.set_title("NUTS")
    ax_nuts.scatter(xs[ys == 0, 0], xs[ys == 0, 1], s=2)
    ax_nuts.scatter(xs[ys == 1, 0], xs[ys == 1, 1], s=2)
    grid = np.zeros_like(x1_plot)
    for w_sample in mcmc.get_samples()['w']:
        wi_1, wi_2 = w_sample
        grid += (wi_1 * x1_plot + wi_2 * x2_plot > 0)
    cm = plt.cm.get_cmap('viridis')
    ax_nuts.pcolormesh(x1_plot, x2_plot, grid, cmap=cm, alpha=0.3)
    pic_i += 1

    fig.savefig('figs/bayes_logistic')
