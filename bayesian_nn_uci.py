import pandas as pd

from bayesian_nn import train_svgd, train_nuts, metrics
from svgd_original_nn import train_original_svgd
from bbb.bbb_training import train_bbb
from pbp.pbp_training import train_pbp
from data_uci.uci import load_uci_dataset, UCI_SETS

MINIBATCHES = 100


def _hidden_dim_for_dataset(d):
    return 100 if d == 'protein' else 50


def _repeats_for_dataset(d):
    # return 2
    return 5 if d == 'protein' else 10


def _epochs_for_dataset(d):
    return 50 if d in ['kin8nm', 'naval', 'power', 'protein'] else 200


def _run_svgd(xs_train, ys_train, xs_test, dset):
    return train_svgd(xs_train,
                      ys_train,
                      xs_test,
                      hidden_dim=_hidden_dim_for_dataset(dset),
                      num_particles=20,
                      num_steps=_epochs_for_dataset(dset) * xs_train.shape[0] // MINIBATCHES,
                      subsample_size=MINIBATCHES)


def _run_original_svgd(xs_train, ys_train, xs_test, dset):
    return train_original_svgd(xs_train,
                               ys_train,
                               xs_test,
                               hidden_dim=_hidden_dim_for_dataset(dset),
                               num_particles=20,
                               num_steps=_epochs_for_dataset(dset) * xs_train.shape[0] // MINIBATCHES,
                               subsample_size=MINIBATCHES)


# def _run_bbb(xs_train, ys_train, xs_test, dset):
#     return train_bbb(xs_train,
#                      ys_train,
#                      xs_test,
#                      hidden_dim=_hidden_dim_for_dataset(dset),
#                      num_particles=50,
#                      num_steps=10000,
#                      batch_size=xs_train.shape[0],
#                      lr=0.001)


def _run_pbp(xs_train, ys_train, xs_test, dset):
    return train_pbp(xs_train,
                     ys_train,
                     xs_test,
                     hidden_dim=_hidden_dim_for_dataset(dset),
                     num_epochs=_epochs_for_dataset(dset))


# def _run_nuts(xs_train, ys_train, xs_test, dset):
#     return train_nuts(xs_train,
#                       ys_train,
#                       xs_test,
#                       hidden_dim=_hidden_dim_for_dataset(dset),
#                       n_nuts=100)


if __name__ == '__main__':

    fname = './results/bnn_results_uci.csv'

    try:
        df = pd.read_csv(fname)
        results = df.to_dict('records')
        done_cases = {(x['alg'], x['dset'], x['set']) for x in results}
    except (FileNotFoundError, pd.errors.EmptyDataError):
        results = []
        done_cases = set()

    try:
        for dset in UCI_SETS:

            for r in range(_repeats_for_dataset(dset)):

                print(f'{dset}, repeat {r}')

                for (alg, fn) in [
                    ('svgd', _run_svgd),
                    ('svgd_orig', _run_original_svgd),
                    ('pbp', _run_pbp),
                ]:
                    if (alg, dset, r) in done_cases:
                        continue

                    xs_train, ys_train, xs_test, ys_test, y_train_mean, y_train_std = load_uci_dataset(dset, split=r)
                    y_mean, y_std, t = fn(xs_train, ys_train, xs_test, dset)
                    rmse, logp = metrics(ys_test, y_mean, y_std, y_train_mean, y_train_std)
                    print(f'{alg} - time={t:4f}s, rmse={rmse:.4f}, log_likelihood={logp:.4f}')
                    # print(y_mean, y_std)
                    results.append({
                        'alg': alg,
                        'dset': dset,
                        'set': r,
                        'time': t,
                        'rmse': rmse,
                        'logp': logp
                    })
                    done_cases.add((alg, dset, r))

    except KeyboardInterrupt:
        pass
    finally:
        if len(results) > 0:
            df = pd.DataFrame.from_records(results)
            df.sort_values(by=['dset', 'alg', 'set'], inplace=True)
            df.to_csv(fname, index=False)
