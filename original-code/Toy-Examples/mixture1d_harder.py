import numpy as np
import numpy.matlib as nm
from svgd import SVGD
import math
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

class Mixture1d:
    def __init__(self, w1, mu1, sigma1, w2, mu2, sigma2):
        self.name = "gaussian"
        self.w1 = w1
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.w2 = w2
        self.mu2 = mu2
        self.sigma2 = sigma2

    def dlnprob_single(self, x):
        a,b,c,d,e,f = self.w1, self.mu1, self.sigma1, self.w2, self.mu2, self.sigma2
        sqrt = math.sqrt
        pi = math.pi
        E = math.e
        return (-((a * E**(-((-b + x)**2/(2 * c**2))) * (-b + x))/(c**3 * sqrt(2 * pi))) - (
            d * E**(-((-e + x)**2/(2 * f**2))) * (-e + x))/(f**3 * sqrt(2 * pi)))/((
            a * E**(-((-b + x)**2/(2 * c**2))))/(c * sqrt(2 * pi)) + (
            d * E**(-((-e + x)**2/(2 * f**2))))/(f * sqrt(2 * pi)))
    
    def dlnprob(self, points):
        func = lambda x : self.dlnprob_single(x)
        vfunc = np.vectorize(func)
        return vfunc(points)

    def plotpdf(self):
        x = np.linspace(-10, 10, 100)
        curve_0 = norm.pdf(x, self.mu1, self.sigma1)
        curve_1 = norm.pdf(x, self.mu2, self.sigma2)
        curve = self.w1 * curve_0 + self.w2 * curve_1
        iter_pdf = plt.plot(x, curve, lw=2)
    
    def axplotpdf(self, ax):
        x = np.linspace(-14, 14, 100)
        curve_0 = norm.pdf(x, self.mu1, self.sigma1)
        curve_1 = norm.pdf(x, self.mu2, self.sigma2)
        curve = self.w1 * curve_0 + self.w2 * curve_1
        ax.plot(x, curve, lw=2)

def assertions():
    model = Mixture1d(1/3, -2, 1, 2/3, 2, 1)
    almost_eq = lambda  x, y: abs(x - y) < 1e-5
    assert almost_eq(model.dlnprob_single(1), 0.963701)
    assert almost_eq(model.dlnprob_single(2), -0.000670813)
    assert almost_eq(model.dlnprob_single(-1), -0.858653)
    assert almost_eq(model.dlnprob_single(-2), 0.0026819)

if __name__ == '__main__':
    assertions()
    np.random.seed(5432)

    w1 = 1/3
    mu = 2.0
    mu1 = -mu
    mu2 = mu
    w2 = 2/3
    sigma1 = 1.0
    sigma2 = 1.0
    model = Mixture1d(w1, mu1, sigma1, w2, mu2, sigma2)
    
    x0 = np.random.normal(-10,1,[100,1])
    x_after = x0

    step_size = 0.1
    # step_size = 0.25
    check_iters = [0, 50, 75, 100, 150, 500]
    print("check_iters:", check_iters)

    os.system("rm -rf ./output/mixture1d_harder_*.csv")
    os.system("rm -rf ./output/mixture1d_harder_*.png")

    fig = None
    axs = None
    def save_on_check_iter_func2(iter_idx, theta):
        global fig
        global axs
        if fig is None: 
            fig = plt.figure(figsize=(4 * len(check_iters), 4))
            axs = fig.subplots(nrows=1, ncols=len(check_iters), sharex=True, sharey=False)
        if iter_idx in check_iters:
            idx = check_iters.index(iter_idx)
            ax = axs[idx]
            ax.set_title(f'{iter_idx}th Iteration')
            ax.set_xlim([-14, 14])
            ax.set_ylim([0, 0.4])
            
            print(f"svgd ({iter_idx}th iteration): ", np.mean(theta, axis=0))
            # np.savetxt(os.path.join(os.path.dirname(__file__), f"./mixture1d_iter_{iter_idx}.csv"), theta, delimiter=",")

            model.axplotpdf(ax)
            # taken from https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
            kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
            kde.fit(theta)
            theta_mesh = np.linspace(-14, 14, 100)
            logprob = kde.score_samples(theta_mesh[:, None])

            ax.fill_between(theta_mesh, np.exp(logprob), alpha=0.5)
            ax.plot(theta, np.full_like(theta, -0.01), '|k', markeredgewidth=1)


    save_on_check_iter_func = save_on_check_iter_func2
    save_on_check_iter_func(0, x_after)
    x_after = SVGD().update(x0, model.dlnprob, n_iter=check_iters[-1], stepsize=step_size, callback=save_on_check_iter_func)
    if save_on_check_iter_func == save_on_check_iter_func2:
        plt.savefig(f"./output/_mixture1d_all_step{step_size}_mu{mu}_w{round(w1,2)}_{model.name}.png")
