import jax.random
import numpy as np
import matplotlib.pyplot as plt
import tqdm

# import pyro
# import pyro.distributions as dist
# from pyro.optim import Adam
#
# from pyro.infer.svgd import SVGD, RBFSteinKernel
# from pyro.infer.mcmc import MCMC
# from pyro.infer.mcmc.nuts import NUTS
# from pyro.infer import SVI, Trace_ELBO
# from pyro.infer.autoguide import AutoNormal

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
    alpha = numpyro.sample("alpha", dist.Gamma(1., 0.01))
    with numpyro.plate("wgt", 2, dim=-1):
        w = numpyro.sample("w", dist.Normal(np.zeros(2), (1. / alpha) * np.ones(2)))
    with numpyro.plate("data", len(x_data), dim=-1):
        logits = x_data @ w
        numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y_data)


if __name__ == '__main__':

    # dataset
    xs, ys = generate_gmm()

    # for plotting purposes
    steps_pic = [1, 20, 50, 100, 150]
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

    print("SVGD")
    steps_elapsed = 0
    inf_key, pred_key, data_key = random.split(random.PRNGKey(42), 3)
    rng_key, inf_key = random.split(inf_key)

    svgd = SteinVI(
        model=logistic_reg_model,
        guide=AutoNormal(logistic_reg_model),
        kernel_fn=RBFKernel(),
        loss=Trace_ELBO(),
        optim=Adam(0.1),
        num_particles=num_particles
    )

    for step in steps_pic:

        result = svgd.run(rng_key, step - steps_elapsed, xs, ys)
        steps_elapsed = step

        pred = Predictive(
            logistic_reg_model,
            return_sites=['w'],
            guide=svgd.guide,
            params=svgd.get_params(result.state),
            num_samples=num_particles,
            batch_ndims=1,  # stein particle dimension
        )

        ax = axs[pic_i]
        ax.scatter(xs[ys == 0, 0], xs[ys == 0, 1], s=2)
        ax.scatter(xs[ys == 1, 0], xs[ys == 1, 1], s=2)

        ws = pred(pred_key, xs)['w']
        for w in ws[0]:
            wi_0, wi_1 = w
            ax.plot(x0_sp, -wi_0 * x0_sp / wi_1, color='black', alpha=min(1, 20. / n))

        ax.set_title(f'Iteration {step}')
        pic_i += 1

    # NUTS ALGORITHM
    print("NUTS")

    n_nuts = 1000
    inf_key, pred_key, data_key = random.split(random.PRNGKey(42), 3)
    rng_key, inf_key = random.split(inf_key)

    nuts_kernel = NUTS(logistic_reg_model, adapt_step_size=True)
    mcmc = MCMC(nuts_kernel, num_samples=n_nuts, num_warmup=n_nuts)
    mcmc.run(rng_key, xs, ys)

    ax_nuts = axs[pic_i]
    ax_nuts.set_title("NUTS")
    ax_nuts.scatter(xs[ys == 0, 0], xs[ys == 0, 1], s=2)
    ax_nuts.scatter(xs[ys == 1, 0], xs[ys == 1, 1], s=2)
    for w_sample in mcmc.get_samples()['w']:
        wi_0, wi_1 = w_sample
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


# from jax import random
# import jax.numpy as jnp
# import numpyro
# import numpyro.distributions as dist
# from numpyro.distributions import constraints
# from numpyro.infer import Predictive, SVI, Trace_ELBO
#
# def model(data):
#     f = numpyro.sample("latent_fairness", dist.Beta(10, 10))
#     with numpyro.plate("N", data.shape[0]):
#         numpyro.sample("obs", dist.Bernoulli(f), obs=data)
#
# def guide(data):
#     alpha_q = numpyro.param("alpha_q", 15., constraint=constraints.positive)
#     beta_q = numpyro.param("beta_q", lambda rng_key: random.exponential(rng_key),
#                            constraint=constraints.positive)
#     numpyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))
#
# data = jnp.concatenate([jnp.ones(6), jnp.zeros(4)])
# optimizer = numpyro.optim.Adam(step_size=0.0005)
# svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
# svi_result = svi.run(random.PRNGKey(0), 2000, data)
# params = svi_result.params
# inferred_mean = params["alpha_q"] / (params["alpha_q"] + params["beta_q"])
# # get posterior samples
# predictive = Predictive(guide, params=params, num_samples=1000)
# samples = predictive(random.PRNGKey(1), data)
# print(samples)
