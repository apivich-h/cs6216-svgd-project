import numpy as np
import scipy.io
import os

MAT = scipy.io.loadmat(f'{os.path.abspath(os.path.dirname(__file__))}/benchmarks.mat')
DSETS = ['banana', 'image', 'ringnorm', 'splice', 'titanic', 'twonorm', 'waveform']


def read_benchmark(name, realisation):
    """Dataset from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html"""
    xs, ys, ids_train, ids_test = MAT[name][0][0]
    ys = (ys == 1).astype(np.int32).flatten()
    real_train = ids_train[realisation] - 1
    real_test = ids_test[realisation] - 1
    return xs[real_train], ys[real_train], xs[real_test], ys[real_test]
