import numpy as np
import numpy.matlib as nm
from svgd import SVGD
import math
import os

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
        return (-d*(-e + x)/e**((-e + x)**2/((2*f**2)))/(f**3*sqrt(2*pi)) - a*(-b + x)/e**((-b + x)**2/((2*c**2)))/(c**3*sqrt(2*pi)))/((a/e**((-b + x)**2/((2*c**2))))/((c*sqrt(2*pi))) + 1/(d*e**((-e + x)**2/((2*f**2)))*((f*sqrt(2*pi)))))
    
    def dlnprob(self, points):
        func = lambda x : self.dlnprob_single(x)
        vfunc = np.vectorize(func)
        return vfunc(points)
    
if __name__ == '__main__':
    w1 = 1/3
    mu1 = -2.0
    mu2 = 2.0
    w2 = 2/3
    sigma1 = 1.0
    sigma2 = 1.0
    model = Mixture1d(w1, mu1, sigma1, w2, mu2, sigma2)
    
    x0 = np.random.normal(-10,1,[1000,1])
    xafter = SVGD().update(x0, model.dlnprob, n_iter=4000, stepsize=0.01)
    
    print("svgd: ", np.mean(xafter,axis=0))
    np.savetxt(os.path.join(os.path.dirname(__file__), "mixture1d.csv"), xafter, delimiter=",")
    print("Copy csv data to here for histogram: https://statscharts.com/bar/histogram?status=edit")