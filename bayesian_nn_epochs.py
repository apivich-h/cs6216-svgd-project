import numpy as np
import pandas as pd
from functools import partial

from jax import random
from numpyro.optim import Adagrad
from numpyro.infer import Predictive, Trace_ELBO, RenyiELBO, init_to_uniform
from numpyro.infer.autoguide import AutoDelta

from numpyro.contrib.einstein.steinvi import SteinVI
from numpyro.contrib.einstein.kernels import RBFKernel

from bayesian_nn import metrics, generate_bnn_model, train_svgd
from svgd_original_nn import train_original_svgd
from pbp.pbp_net import PBPNet
from data_uci.uci import load_uci_dataset, UCI_SETS

MINIBATCHES = 100
EPOCHS_TO_RUN = list(range(1, 21)) + list(range(25, 51, 5)) + list(range(60, 101, 10))


def _hidden_dim_for_dataset(d):
    return 100 if d == 'protein' else 50


def _run_svgd_incrementally(xs_train, ys_train, xs_test, dset):
    for ep in EPOCHS_TO_RUN:
        y_mean, y_std, _ = train_svgd(xs_train,
                                      ys_train,
                                      xs_test,
                                      hidden_dim=_hidden_dim_for_dataset(dset),
                                      num_particles=20,
                                      num_steps=ep * xs_train.shape[0] // MINIBATCHES,
                                      subsample_size=MINIBATCHES,
                                      lr=1e-1)
        yield ep, y_mean, y_std


def _run_original_svgd_incrementally(xs_train, ys_train, xs_test, dset):
    for ep in EPOCHS_TO_RUN:
        y_mean, y_std, _ = train_original_svgd(xs_train,
                                               ys_train,
                                               xs_test,
                                               hidden_dim=_hidden_dim_for_dataset(dset),
                                               num_particles=20,
                                               num_steps=ep * xs_train.shape[0] // MINIBATCHES,
                                               subsample_size=MINIBATCHES,
                                               lr=1e-3)
        yield ep, y_mean, y_std


def _run_pbp_incrementally(xs_train, ys_train, xs_test, dset):
    net = PBPNet(xs_train, ys_train, [_hidden_dim_for_dataset(dset)], normalize=False)

    curr_epochs = 0

    for ep in EPOCHS_TO_RUN:
        d_epoch = ep - curr_epochs
        net.train(xs_train, ys_train, n_epochs=d_epoch)
        mean, var, noise_var = net.predict(xs_test)
        curr_epochs = ep
        yield ep, mean, np.sqrt(var + noise_var)


if __name__ == '__main__':

    fname = './results/bnn_results_epochs.csv'
    try:
        df = pd.read_csv(fname)
        results = df.to_dict('records')
        done_cases = {(x['Algorithm'], x['Dataset'], x['Repeat']) for x in results}
    except (FileNotFoundError, pd.errors.EmptyDataError):
        results = []
        done_cases = set()

    try:
        for dset, r in [
            ('boston', 0),
            ('kin8nm', 0),
            ('protein', 0),
            ('yacht', 0)
        ]:

            print(f'{dset}, repeat {r}')

            for (alg, fn) in [
                ('SVGD (NumPyro)', _run_svgd_incrementally),
                ('SVGD (original repo)', _run_original_svgd_incrementally),
                ('PBP', _run_pbp_incrementally),
            ]:
                if (alg, dset, r) in done_cases:
                    continue

                xs_train, ys_train, xs_test, ys_test, y_train_mean, y_train_std = load_uci_dataset(dset, split=r)

                for (ep, y_mean, y_std) in fn(xs_train, ys_train, xs_test, dset):
                    rmse, logp = metrics(ys_test, y_mean, y_std, y_train_mean, y_train_std)
                    print(f'{alg}, epoch={ep} - rmse={rmse:.4f}, log_likelihood={logp:.4f}')
                    results.append({
                        'Algorithm': alg,
                        'Dataset': dset,
                        'Repeat': r,
                        'Epoch': ep,
                        'RMSE': rmse,
                        'Log likelihood': logp
                    })
                done_cases.add((alg, dset, r))

    except KeyboardInterrupt:
        pass
    finally:
        if len(results) > 0:
            pd.DataFrame.from_records(results).to_csv(fname, index=False)
