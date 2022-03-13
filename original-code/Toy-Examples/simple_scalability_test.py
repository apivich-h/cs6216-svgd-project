import numpy as np
import numpy.matlib as nm
from svgd import SVGD
import timeit
import sys
import matplotlib.pyplot as plt

class MVN:
    def __init__(self, mu, A):
        self.mu = mu
        self.A = A
    
    def dlnprob(self, theta):
        return -1*np.matmul(theta-nm.repmat(self.mu, theta.shape[0], 1), self.A)
    

timing_particle_sizes = [100, 141, 200, 283, 400, 566, 800, 1131, 1600]
if "redo_timing" in sys.argv:
    A = np.array([[0.2260,0.1652],[0.1652,0.6779]])
    mu = np.array([-0.6871,0.8010])
    
    model = MVN(mu, A)
    
    for i in range(50):
      time_row = []
      for N in timing_particle_sizes:
        x0 = np.random.normal(0,1, [N,2])
        time1 = timeit.default_timer()
        theta = SVGD().update(x0, model.dlnprob, n_iter=1000, stepsize=0.01)
        time2 = timeit.default_timer()
        time_row.append(time2 - time1)
        print("#particle - time:", N, "\t", time2 - time1)
        print("svgd - ground_truth: ", np.mean(theta,axis=0) - mu)
      with open("./simple_scalability_test_timing.csv", 'a') as f:
        line = ",".join([str(x) for x in time_row]) + "\n"
        print("writeline:", line)
        f.write(line)


timing_data = np.loadtxt("./simple_scalability_test_timing.csv", delimiter=",")
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
plt.savefig(f"../../figs/toy-timing-particles.png")