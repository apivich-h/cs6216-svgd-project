import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import time
from functools import partial

from bayesian_nn import train_svgd, metrics
from data_uci.uci import load_uci_dataset, UCI_SETS


if __name__ == '__main__':

    for d in UCI_SETS:
        xs_train, ys_train, xs_test, ys_test, y_train_mean, y_train_std = load_uci_dataset(d, split=0)
        print(d, xs_train.shape, ys_train.shape, xs_test.shape, ys_test.shape)

        y_mean, y_std, t = train_svgd(xs_train, ys_train, xs_test, hidden_dim=50, num_particles=20)
        print(metrics(ys_test, y_mean, y_std, y_train_mean, y_train_std))
