import time
import numpy as np

from pbp.pbp_net import PBPNet


def train_pbp(xs_train, ys_train, xs_test, hidden_dim=50, num_epochs=40):

    t = time.time()
    net = PBPNet(xs_train, ys_train, [hidden_dim], normalize=False)
    net.train(xs_train, ys_train, n_epochs=num_epochs)
    t = time.time() - t

    mean, var, noise_var = net.predict(xs_test)

    return mean, np.sqrt(var + noise_var), t
