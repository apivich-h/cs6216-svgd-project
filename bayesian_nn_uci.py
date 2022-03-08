import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import time
from functools import partial

from bayesian_nn import train_svgd, train_nuts, metrics
from bbb.bbb_training import train_bbb
from data_uci.uci import load_uci_dataset, UCI_SETS

MINIBATCHES = 200


def _hidden_dim_for_dataset(d):
    return 100 if d == 'protein' else 50


def _repeats_for_dataset(d):
    return 1
    # return 5 if d == 'protein' else 20


def _run_svgd(xs_train, ys_train, xs_test, dset):
    return train_svgd(xs_train,
                      ys_train,
                      xs_test,
                      hidden_dim=_hidden_dim_for_dataset(dset),
                      num_particles=20,
                      num_steps=1000 * xs_train.shape[0] // MINIBATCHES,
                      subsample_size=MINIBATCHES)


def _run_bbb(xs_train, ys_train, xs_test, dset):
    return train_bbb(xs_train,
                     ys_train,
                     xs_test,
                     hidden_dim=_hidden_dim_for_dataset(dset),
                     num_particles=50,
                     num_epochs=10000,
                     batch_size=xs_train.shape[0],
                     lr=0.001)


def _run_nuts(xs_train, ys_train, xs_test, dset):
    return train_nuts(xs_train,
                      ys_train,
                      xs_test,
                      hidden_dim=_hidden_dim_for_dataset(dset),
                      n_nuts=100)


if __name__ == '__main__':

    results = []

    try:
        for dset in UCI_SETS:

            for r in range(_repeats_for_dataset(dset)):

                print(f'{dset}, repeat {r}')

                for (alg, fn) in [('svgd', _run_svgd),
                                  ('bbb', _run_bbb),
                                  # ('nuts', _run_nuts)
                                  ]:

                    xs_train, ys_train, xs_test, ys_test, y_train_mean, y_train_std = load_uci_dataset(dset, split=r)
                    y_mean, y_std, t = fn(xs_train, ys_train, xs_test, dset)
                    rmse, logp = metrics(ys_test, y_mean, y_std, y_train_mean, y_train_std)

                    results.append({
                        'alg': alg,
                        'dset': dset,
                        'set': r,
                        'time': t,
                        'rmse': rmse,
                        'logp': logp
                    })
    except KeyboardInterrupt:
        pass
    finally:
        if len(results) > 0:
            pd.DataFrame.from_records(results).to_csv('results/bnn_results_uci.csv')
