import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm
import pandas as pd
from scipy import stats
from functools import partial
import time

from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.optim import Adam, Adagrad
from numpyro.infer import Predictive, Trace_ELBO, init_to_uniform, init_to_feasible, init_to_value
from numpyro.infer.autoguide import AutoNormal, AutoDelta

from numpyro.contrib.einstein.steinvi import SteinVI
from numpyro.contrib.einstein.kernels import RBFKernel
from numpyro.infer.mcmc import MCMC
from numpyro.infer.hmc import NUTS, HMC

"""
A lot of code on this page was taken from https://num.pyro.ai/en/latest/examples/stein_bnn.html
"""


def generate_data():
    x = np.linspace(0, 0.5, 1000)
    eps = 0.02 * np.random.randn(x.shape[0])
    y = x - 0.2 * x ** 2 + 0.3 * np.sin(2 * np.pi * (x + eps)) + 0.3 * np.sin(4 * np.pi * (x + eps)) + eps
    return x.reshape(-1, 1), y


def generate_bnn_model(hidden_dim=50):

    def model(x, y=None, subsample_size=100):
        prec_nn = numpyro.sample(
            "prec_nn", dist.Gamma(1.0, 0.1)
        )  # hyper prior for precision of nn weights and biases

        n, m = x.shape

        with numpyro.plate("l1_hidden", hidden_dim, dim=-1):
            # prior l1 bias term
            b1 = numpyro.sample(
                "nn_b1",
                dist.Normal(
                    0.0,
                    1.0 / jnp.sqrt(prec_nn),
                ),
            )
            assert b1.shape == (hidden_dim,)

            with numpyro.plate("l1_feat", m, dim=-2):
                w1 = numpyro.sample(
                    "nn_w1", dist.Normal(0.0, 1.0 / jnp.sqrt(prec_nn))
                )  # prior on l1 weights
                assert w1.shape == (m, hidden_dim)

        with numpyro.plate("l2_hidden", hidden_dim, dim=-1):
            w2 = numpyro.sample(
                "nn_w2", dist.Normal(0.0, 1.0 / jnp.sqrt(prec_nn))
            )  # prior on output weights

        b2 = numpyro.sample(
            "nn_b2", dist.Normal(0.0, 1.0 / jnp.sqrt(prec_nn))
        )  # prior on output bias term

        # precision prior on observations
        prec_obs = numpyro.sample("prec_obs", dist.Gamma(1.0, 0.1))
        with numpyro.plate(
                "data_covertype",
                x.shape[0],
                subsample_size=subsample_size,
                dim=-1,
        ):
            batch_x = numpyro.subsample(x, event_dim=1)
            if y is not None:
                batch_y = numpyro.subsample(y, event_dim=0)
            else:
                batch_y = y

            numpyro.sample(
                "y",
                dist.Normal(
                    jnp.maximum(batch_x @ w1 + b1, 0) @ w2 + b2, 1.0 / jnp.sqrt(prec_obs)
                ),  # h = x * w1 + b1; y = relu(h) * w2 + b2
                obs=batch_y,
            )

    return model


def metrics(ys_test, ys_pred_mean, ys_pred_std, y_data_mean=None, y_data_std=None):
    if y_data_mean is not None and y_data_std is not None:
        ys_test = ys_test * y_data_std + y_data_mean
        ys_pred_mean = ys_pred_mean * y_data_std + y_data_mean
        ys_pred_std = ys_pred_std * y_data_std
    rmse = np.sqrt(np.mean((ys_test - ys_pred_mean) ** 2))
    logp = np.log(stats.norm.pdf(ys_test, loc=ys_pred_mean, scale=ys_pred_std).mean())
    return rmse, logp


def train_svgd(xs_train, ys_train, xs_test, hidden_dim=50, num_particles=20, num_steps=5000):
    bnn_model = generate_bnn_model(hidden_dim=hidden_dim)

    inf_key, pred_key, data_key = random.split(random.PRNGKey(42), 3)
    rng_key, inf_key = random.split(inf_key)

    svgd = SteinVI(
        model=bnn_model,
        guide=AutoDelta(bnn_model, init_loc_fn=partial(init_to_uniform, radius=0.1)),
        kernel_fn=RBFKernel(),
        loss=Trace_ELBO(),
        optim=Adagrad(1e-3),
        num_particles=num_particles
    )

    t = time.time()
    result = svgd.run(rng_key, num_steps, xs_train, ys_train)
    t = time.time() - t

    predictive = Predictive(bnn_model,
                            guide=svgd.guide,
                            params=svgd.get_params(result.state),
                            num_samples=1,
                            batch_ndims=1)

    preds = predictive(pred_key, xs_test, subsample_size=xs_test.shape[0])
    y_pred = preds['y'][0]
    y_mean = np.mean(y_pred, axis=0)
    y_std = np.std(y_pred, axis=0)
    return y_mean, y_std, t


if __name__ == '__main__':
    xs, ys = generate_data()
    x_test = np.linspace(-0.5, 1, 200).reshape(-1, 1)

    print("SVGD")
    y_mean, y_std, t = train_svgd(xs, ys, x_test, hidden_dim=50, num_particles=100)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs.flatten(), ys, 'o', markersize=1)
    ax.plot(x_test.flatten(), y_mean)
    ax.fill_between(x_test, y_mean - y_std, y_mean + y_std, alpha=0.2)

    fig.savefig('figs/bayesian_nn_svi')