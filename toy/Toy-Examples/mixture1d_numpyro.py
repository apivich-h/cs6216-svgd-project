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

from sklearn.neighbors import KernelDensity

shift = 10.

from jax import lax
import jax.numpy as jnp

def validate_sample(log_prob_fn):
    def wrapper(self, *args, **kwargs):
        log_prob = log_prob_fn(self, *args, *kwargs)
        if self._validate_args:
            value = kwargs["value"] if "value" in kwargs else args[0]
            mask = self._validate_sample(value)
            log_prob = jnp.where(mask, log_prob, -jnp.inf)
        return log_prob
    return wrapper

def _reshape(x, shape):
    if isinstance(x, (int, float, np.ndarray, np.generic)):
        return np.reshape(x, shape)
    else:
        return jnp.reshape(x, shape)

def promote_shapes(*args, shape=()):
    # adapted from lax.lax_numpy
    if len(args) < 2 and not shape:
        return args
    else:
        shapes = [jnp.shape(arg) for arg in args]
        num_dims = len(lax.broadcast_shapes(shape, *shapes))
        return [
            _reshape(arg, (1,) * (num_dims - len(s)) + s) if len(s) < num_dims else arg
            for arg, s in zip(args, shapes)
        ]

class GaussianMixture1d(dist.Distribution):
    arg_constraints = {
        "loc1": dist.constraints.real, 
        "scale1": dist.constraints.positive,
        "weight1": dist.constraints.real, 
        "loc2": dist.constraints.real, 
        "scale2": dist.constraints.positive,
    }
    support = dist.constraints.real
    reparametrized_params = ["loc1", "scale1", "weight1", "loc2", "scale2"]

    def __init__(self, loc1=-2.0, scale1=1.0, weight1=1/3, loc2=2.0, scale2=1.0, validate_args=None):
        self.loc1, self.scale1, self.weight1, self.loc2, self.scale2 = \
            promote_shapes(loc1, scale1, weight1, loc2, scale2)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(loc1), jnp.shape(scale1), jnp.shape(weight1),
            jnp.shape(loc2), jnp.shape(scale2))
        super(GaussianMixture1d, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        choose = random.bernoulli(key, p=np.float32(self.weight1), shape=sample_shape + self.batch_shape + self.event_shape)
        choose = choose.astype("float32")
        revchoose = 1 - choose
        stacked_choices = jnp.stack([choose, revchoose])
        eps1 = random.normal(
            key, shape=sample_shape + self.batch_shape + self.event_shape)
        eps1 = self.loc1 + eps1 * self.scale1
        eps2 = random.normal(
            key, shape=sample_shape + self.batch_shape + self.event_shape)
        eps2 = self.loc2 + eps2 * self.scale2
        stacked_eps = jnp.stack([eps1, eps2])
        return jnp.multiply(stacked_choices, stacked_eps)

    @validate_sample
    def log_prob(self, value):
        value_scaled1 = (value - self.loc1) / self.scale1
        term1 = jnp.exp(-0.5 * value_scaled1**2) / (jnp.sqrt(2 * jnp.pi) * self.scale1)
        value_scaled2 = (value - self.loc2) / self.scale2
        term2 = jnp.exp(-0.5 * value_scaled2**2) / (jnp.sqrt(2 * jnp.pi) * self.scale2)
        return jnp.log(self.weight1 * term1 + (1 - self.weight1) * term2)

    # @validate_sample
    # def log_prob(self, value):
    #     normalize_term = jnp.log(jnp.sqrt(2 * jnp.pi) * self.scale1)
    #     value_scaled = (value - self.loc1) / self.scale1
    #     return -0.5 * value_scaled**2 - normalize_term

def axplotpdf(ax, mu1, sigma1, mu2, sigma2, w1, w2):
    x = np.linspace(-14, 14, 100)
    curve_0 = stats.norm.pdf(x, mu1, sigma1)
    curve_1 = stats.norm.pdf(x, mu2, sigma2)
    curve = w1 * curve_0 + w2 * curve_1
    ax.plot(x, curve, color='orange', lw=2)

def multimodal_model(obs=None):
    # generate multi-peak Normal distribution
    # to determine peak we use beta distribution (so that the term is differentiable)
    # z = numpyro.sample("z", dist.Beta(0.002, 0.001))
    mu = numpyro.sample("mu", GaussianMixture1d(-2, 1, 1/3, 2, 1))
    with numpyro.plate("data", size=len(obs)):
        numpyro.sample("obs", dist.Uniform(-20, 20), obs=0.5)


if __name__ == '__main__':

    PARTICLE_SIZE = 100
    fake_data = np.array([[0.5]])
    steps_pic = [1, 10, 25, 50, 75, 100] # 150] #, 500]

    fig = plt.figure(figsize=(4 * len(steps_pic), 4))
    axs = fig.subplots(nrows=1, ncols=len(steps_pic), sharex=True, sharey=True)
    xs = np.linspace(-6, 6, 200)

    inf_key, pred_key, data_key = random.split(random.PRNGKey(42), 3)
    rng_key, inf_key = random.split(inf_key)
    
    def manual_guide(guess=-10):
        numpyro.sample("mu", dist.Normal(-10, 1))

    USING_AUTO_GUIDE = True

    if USING_AUTO_GUIDE:
        guide = AutoNormal(multimodal_model)
    else:
        guide = manual_guide # doesn't work
    

    svgd = SteinVI(model=multimodal_model,
                   guide=guide, # AutoNormal(multimodal_model)
                   optim=Adam(step_size=0.1),
                   loss=Trace_ELBO(PARTICLE_SIZE),
                   kernel_fn=RBFKernel(),
                   num_particles=PARTICLE_SIZE)

    steps_elapsed = 0

    for (ax, step) in zip(axs, steps_pic):

        result = svgd.run(rng_key, step, fake_data)
        steps_elapsed = step

        pred = Predictive(
            multimodal_model,
            return_sites=['mu'],
            guide=svgd.guide,
            params=svgd.get_params(result.state),
            num_samples=1,
            batch_ndims=1,  # stein particle dimension
        )
        ax.set_ylim([0, 0.4])
        axplotpdf(ax, -2, 1, 2, 1, 1/3, 2/3)

        mu = pred(pred_key, fake_data)['mu']
        mu_flat = np.array(mu.flatten())
        mu_arr = mu_flat.reshape(PARTICLE_SIZE, 1)
        kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
        kde.fit(mu_arr)
        theta_mesh = np.linspace(-14, 14, 100)
        logprob = kde.score_samples(theta_mesh[:, None])
        ax.fill_between(theta_mesh, np.exp(logprob), color='orange', alpha=0.5)
        ax.plot(mu_arr, np.full_like(mu_flat, -0.01), '|k', color='orange',  markeredgewidth=1)


        # ax.hist(mu.flatten(), bins=50)
        
        ax.set_title(f'{step} iterations')

    fig.savefig('../../figs/toy-figure1-numpyro.png')
