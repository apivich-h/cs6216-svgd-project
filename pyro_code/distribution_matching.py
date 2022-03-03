import matplotlib.pyplot as plt
import tqdm
from scipy import stats
import numpy as np

import torch
import pyro
import pyro.distributions as dist
from pyro.optim import Adam

from pyro.infer.svgd import SVGD, RBFSteinKernel

shift = 10.


def p(x):
    return (1 / 3) * stats.norm(shift - 2., 1.).pdf(x) + (2 / 3) * stats.norm(shift + 2., 1.).pdf(x)


def multimodal_model():
    # generate multi-peak Normal distribution
    # to determine peak we use beta distribution (so that the term is differentiable)
    z = pyro.sample("z", dist.Beta(0.002, 0.001))
    return pyro.sample("mu", dist.Normal(torch.Tensor([shift - 2.]) + 4. * z, torch.Tensor([1.])))


if __name__ == '__main__':

    num_particles = 1000
    steps_pic = [1, 25, 50, 75, 100, 150]

    pyro.clear_param_store()

    kernel = RBFSteinKernel()
    adam = Adam({"lr": 0.1})
    svgd = SVGD(multimodal_model, kernel, adam, num_particles=num_particles, max_plate_nesting=1)

    fig = plt.figure(figsize=(4 * len(steps_pic), 4))
    axs = fig.subplots(nrows=1, ncols=len(steps_pic), sharex=True, sharey=True)
    pic_i = 0
    xs = np.linspace(shift - 4., shift + 4., 200)

    for i in tqdm.trange(steps_pic[-1] + 1):
        svgd.step()
        if i in steps_pic:
            particles = svgd.get_named_particles()['mu'].detach().numpy()
            ax = axs[pic_i]
            ax.hist(particles, bins=50, density=True)
            ax.plot(xs, p(xs))
            ax.set_title(f'Iteration {i}')
            pic_i += 1

    fig.savefig('figs/dist_matching_svgd')
