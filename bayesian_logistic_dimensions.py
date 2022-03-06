import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

from bayesian_logistic import generate_gmm, logistic_reg_model_generator, metrics


def train_svgd(xs_train, ys_train, xs_test, ys_test, num_particles=100, steps=100):
    dim = xs_train.shape[1]
    model = logistic_reg_model_generator(dim, bias=False)

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

    t = time.time()
    result = svgd.run(rng_key, steps, xs_train, ys_train)
    t = time.time() - t

    pred = Predictive(
        model,
        return_sites=['w'],
        guide=svgd.guide,
        params=svgd.get_params(result.state),
        num_samples=num_particles,
        batch_ndims=1,  # stein particle dimension
    )

    samples = pred(pred_key, xs_test)
    ws = samples['w'][0]
    ys_pred_logit = xs_test @ ws.T
    ys_pred = (1 / (1 + np.exp(-ys_pred_logit))).mean(axis=1)

    acc, abs_acc, logp = metrics(ys_test, ys_pred)
    return acc, abs_acc, logp, t


def train_nuts(xs_train, ys_train, xs_test, ys_test):
    dim = xs_train.shape[1]
    model = logistic_reg_model_generator(dim, bias=False)

    n_nuts = 1000

    inf_key, pred_key, data_key = random.split(random.PRNGKey(42), 3)
    rng_key, inf_key = random.split(inf_key)

    nuts_kernel = NUTS(model, adapt_step_size=True)
    mcmc = MCMC(nuts_kernel, num_samples=n_nuts, num_warmup=n_nuts)

    t = time.time()
    mcmc.run(rng_key, xs_train, ys_train)
    t = time.time() - t

    samples = mcmc.get_samples()
    ws = samples['w']
    ys_pred_logit = xs_test @ ws.T
    ys_pred = (1 / (1 + np.exp(-ys_pred_logit))).mean(axis=1)

    acc, abs_acc, logp = metrics(ys_test, ys_pred)
    return acc, abs_acc, logp, t


if __name__ == '__main__':

    dimensions = [2, 4, 8, 16, 32, 64, 128, 256]

    results = []
    for dim in dimensions:

        for i in range(5):
            xs_tr, ys_tr = generate_gmm(dims=dim, count=500, seed=42 + i)
            xs_te, ys_te = generate_gmm(dims=dim, count=500, seed=24 + i)

            acc, abs_acc, logp, t = train_svgd(xs_tr, ys_tr, xs_te, ys_te)
            results.append({
                'alg': 'svgd',
                'dim': dim,
                'set': i,
                'time': t,
                'acc': acc,
                'abs_acc': abs_acc,
                'logp': logp
            })

            acc, abs_acc, logp, t = train_nuts(xs_tr, ys_tr, xs_te, ys_te)
            results.append({
                'alg': 'nuts',
                'dim': dim,
                'set': i,
                'time': t,
                'acc': acc,
                'abs_acc': abs_acc,
                'logp': logp
            })

    pd.DataFrame.from_records(results).to_csv('results/logstic_results_dims.csv')
