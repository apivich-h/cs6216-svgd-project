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
import sys
import timeit

class MVN:
    def __init__(self, mu, A):
        self.mu = mu
        self.A = A
    
    def p(self, theta):
        return stats.multivariate_normal(theta, self.mu, self.A)

A = np.array([[0.2260,0.1652],[0.1652,0.6779]])
mu = np.array([-0.6871,0.8010])
    
model = MVN(mu, A)

def multinormal_model(obs=None):
    # generate multi-peak Normal distribution
    # to determine peak we use beta distribution (so that the term is differentiable)
    # z = numpyro.sample("z", dist.Beta(0.002, 0.001))
    mu = numpyro.sample("mu", dist.MultivariateNormal(loc=model.mu, covariance_matrix=model.A))
    with numpyro.plate("data", size=len(obs)):
        numpyro.sample("obs", dist.Uniform(-20, 20), obs=0.5)


timing_particle_sizes = [100, 141, 200, 283, 400, 566, 800, 1131, 1600]
if "redo_timing" in sys.argv:
    
    fake_data = np.array([[0.5]])

    inf_key, pred_key, data_key = random.split(random.PRNGKey(42), 3)
    rng_key, inf_key = random.split(inf_key)

    for i in range(50):
      time_row = []
      for N in timing_particle_sizes:
        guide = AutoNormal(multinormal_model)
        svgd = SteinVI(model=multinormal_model,
              guide=guide, # AutoNormal(multimodal_model)
              optim=Adam(step_size=0.1),
              loss=Trace_ELBO(N),
              kernel_fn=RBFKernel(),
              num_particles=N)
        
        time1 = timeit.default_timer()
        print("start timer:", time1)

        result = svgd.run(rng_key, 1000, fake_data)
        print("after run.")
        pred = Predictive(
            multinormal_model,
            return_sites=['mu'],
            guide=svgd.guide,
            params=svgd.get_params(result.state),
            num_samples=1,
            batch_ndims=1,  # stein particle dimension
        )
        time2 = timeit.default_timer()
        time_row.append(time2 - time1)
        print("#particle - time:", N, "\t", time2 - time1)
        muemp = pred(pred_key, fake_data)['mu']
        mu_flat = np.array(muemp.flatten())
        mu_arr = mu_flat.reshape(N, 2)
        print("svgd - ground_truth: ", np.mean(mu_arr, axis=0) - mu)
      with open("./simple_scalability_test_timing_numpyro_elbo.csv", 'a') as f:
        line = ",".join([str(x) for x in time_row]) + "\n"
        print("writeline:", line)
        f.write(line)


timing_data = np.loadtxt("./simple_scalability_test_timing_numpyro_elbo.csv", delimiter=",")
dropped_timing_data = timing_data[1:]
mean_data = np.mean(dropped_timing_data, axis=0)
print(mean_data)
div100_particle_sizes = np.array(timing_particle_sizes)/100
fit_coefficients = np.polyfit(div100_particle_sizes, mean_data, 2)
print(fit_coefficients)
fit_func = np.poly1d(fit_coefficients)

plt.clf()
mesh = np.linspace(0, timing_particle_sizes[-1]/100, 100)
plt.xlabel("#particles")
plt.ylabel("time(s)")
plt.plot(mesh * 100, [fit_func(x) for x in mesh], '--')
plt.plot(timing_particle_sizes, mean_data, "ro", linestyle="None",markersize=6)
plt.legend([f"${round(fit_coefficients[0], 3)} x'^2 + {round(fit_coefficients[1], 3)} x' + {round(fit_coefficients[2],3)}$  (x' = x/100)", "averaged timing data"])
plt.savefig(f"../../figs/toy-timing-particles-numpyro-elbo.png")