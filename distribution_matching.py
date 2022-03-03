import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.optim import Adam
from numpyro.infer import Predictive, Trace_ELBO, init_to_uniform, init_to_feasible, init_to_value
from numpyro.infer.autoguide import AutoNormal

from numpyro.contrib.einstein.steinvi import SteinVI
from numpyro.contrib.einstein.kernels import RBFKernel

shift = 10.


def p(x):
    return stats.norm.pdf(x, loc=0., scale=1.)


def data():
    return np.random.normal(loc=0., scale=1., size=(1000,))


def multimodal_model(obs=None):
    # generate multi-peak Normal distribution
    # to determine peak we use beta distribution (so that the term is differentiable)
    # z = numpyro.sample("z", dist.Beta(0.002, 0.001))
    mu = numpyro.sample("mu", dist.Normal(np.array([-10.]), np.array([1.])))
    with numpyro.plate("data", size=len(obs)):
        numpyro.sample("obs", dist.Normal(mu, np.array([1.])), obs=obs)


def guide():
    numpyro.param("mu", lambda x: dist.Normal(np.array([-10.]), np.array([1.]))())


if __name__ == '__main__':

    x_data = data()
    num_particles = 1000
    steps_pic = [1, 25, 50, 75, 100, 150]

    fig = plt.figure(figsize=(4 * len(steps_pic), 4))
    axs = fig.subplots(nrows=1, ncols=len(steps_pic), sharex=True, sharey=True)
    xs = np.linspace(-6, 6, 200)

    inf_key, pred_key, data_key = random.split(random.PRNGKey(42), 3)
    rng_key, inf_key = random.split(inf_key)

    svgd = SteinVI(model=multimodal_model,
                   guide=AutoNormal(multimodal_model),
                   optim=Adam(step_size=0.1),
                   loss=Trace_ELBO(100),
                   kernel_fn=RBFKernel(),
                   num_particles=num_particles)

    steps_elapsed = 0

    for (ax, step) in zip(axs, steps_pic):

        result = svgd.run(rng_key, step - steps_elapsed, x_data)
        steps_elapsed = step

        pred = Predictive(
            multimodal_model,
            return_sites=['mu'],
            guide=svgd.guide,
            params=svgd.get_params(result.state),
            num_samples=1,
            batch_ndims=1,  # stein particle dimension
        )

        mu = pred(pred_key, x_data)['mu']

        ax.hist(mu.flatten(), bins=50)
        # ax.plot(xs, p(xs))
        ax.set_title(f'{step} iterations')

    fig.savefig('figs/dist_matching_svgd')
