import numpy as np
import pandas as pd

from bayesian_nn import metrics, train_svgd
from svgd_original_nn import train_original_svgd
from pbp.pbp_net import PBPNet
from data_uci.uci import load_uci_dataset, UCI_SETS

MINIBATCHES = 100


def _hidden_dim_for_dataset(d):
    return 100 if d == 'protein' else 50


def _epochs_for_dataset(d):
    if d in ['kin8nm', 'naval', 'power', 'protein']:
        return [1] + list(range(25, 101, 25))
    else:
        return [1] + list(range(25, 100, 25)) + list(range(100, 501, 50))


def _run_svgd_incrementally(xs_train, ys_train, xs_test, dset):
    for ep in _epochs_for_dataset(dset):
        y_mean, y_std, _ = train_svgd(xs_train,
                                      ys_train,
                                      xs_test,
                                      hidden_dim=_hidden_dim_for_dataset(dset),
                                      num_particles=20,
                                      num_steps=ep * xs_train.shape[0] // MINIBATCHES,
                                      subsample_size=MINIBATCHES)
        yield ep, y_mean, y_std


def _run_original_svgd_incrementally(xs_train, ys_train, xs_test, dset):
    for ep in _epochs_for_dataset(dset):
        y_mean, y_std, _ = train_original_svgd(xs_train,
                                               ys_train,
                                               xs_test,
                                               hidden_dim=_hidden_dim_for_dataset(dset),
                                               num_particles=20,
                                               num_steps=ep * xs_train.shape[0] // MINIBATCHES,
                                               subsample_size=MINIBATCHES)
        yield ep, y_mean, y_std


def _run_pbp_incrementally(xs_train, ys_train, xs_test, dset):
    net = PBPNet(xs_train, ys_train, [_hidden_dim_for_dataset(dset)], normalize=False)

    curr_epochs = 0

    for ep in _epochs_for_dataset(dset):
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
            ('energy', 0),
            ('yacht', 0),
            ('naval', 0),
            ('protein', 0),
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
