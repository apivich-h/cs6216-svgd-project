import numpy as np
import numpy.matlib as nm
from svgd import SVGD
import math
import os
import matplotlib.pyplot as plt
from scipy.stats import norm

class Mixture1d:
    def __init__(self, w1, mu1, sigma1, w2, mu2, sigma2):
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

if __name__ == '__main__':
    w1 = 1/3
    mu1 = -2.0
    mu2 = 2.0
    w2 = 2/3
    sigma1 = 1.0
    sigma2 = 1.0
    model = Mixture1d(w1, mu1, sigma1, w2, mu2, sigma2)
    
    x0 = np.random.normal(-10,1,[100,1])
    x_after = x0

    print(f"svgd ({0}th iteration): ", np.mean(x_after,axis=0))
    np.savetxt(os.path.join(os.path.dirname(__file__), f"mixture1d_iter_{0}.csv"), x_after, delimiter=",")
    print("Copy csv data to here for histogram: https://statscharts.com/bar/histogram?status=edit")

    # plot
    plt.clf()
    iter_hist = plt.hist(x_after, 20, density=True, stacked=True)
    model.plotpdf()

    plt.savefig(f"mixture1d_iter_{0}.png")
    plt.show()

    check_iters = [50, 75, 100, 150, 500]

    for i in range(5):
        x_after = SVGD().update(x0, model.dlnprob, n_iter=check_iters[i], stepsize=0.25)
        print(f"svgd ({check_iters[i]}th iteration): ", np.mean(x_after,axis=0))
        np.savetxt(os.path.join(os.path.dirname(__file__), f"mixture1d_iter_{check_iters[i]}.csv"), x_after, delimiter=",")
        print("Copy csv data to here for histogram: https://statscharts.com/bar/histogram?status=edit")

        # plot
        plt.clf()
        iter_hist = plt.hist(x_after, 20, density=True, stacked=True)
        model.plotpdf()

        plt.savefig(f"mixture1d_iter_{check_iters[i]}.png")
        plt.show()

    # xafter = SVGD().update(x0, model.dlnprob, n_iter=4000, stepsize=0.01)
    
    # print("svgd: ", np.mean(xafter,axis=0))
    # np.savetxt(os.path.join(os.path.dirname(__file__), "mixture1d.csv"), xafter, delimiter=",")
    # print("Copy csv data to here for histogram: https://statscharts.com/bar/histogram?status=edit")