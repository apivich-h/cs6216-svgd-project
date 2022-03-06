import os
import numpy as np

"""
This section of the code is adapted from
https://github.com/ratschlab/bnn_priors/tree/main/bnn_priors.

The original code was written for reading data_covertype into a torch-friendly format,
and put into a custom class. 
We use the compiled data_covertype, however then keep the read data_covertype as numpy format.
"""

UCI_SETS = [
    'boston',
    'concrete',
    'energy',
    'kin8nm',
    'naval',
    'power',
    'protein',
    'wine',
    'yacht',
]


def load_uci_dataset(dataset, split, dtype='float32'):
    _ROOT = os.path.abspath(os.path.dirname(__file__))
    dataset_dir = f'{_ROOT}/{dataset}/'
    data = np.loadtxt(f'{dataset_dir}/data.txt').astype(getattr(np, dtype))
    index_features = np.loadtxt(f'{dataset_dir}/index_features.txt')
    index_target = np.loadtxt(f'{dataset_dir}/index_target.txt')
    X_unnorm = data[:, index_features.astype(int)]
    y_unnorm = data[:, index_target.astype(int):index_target.astype(int) + 1]

    # split into train and test
    index_train = np.loadtxt(f'{dataset_dir}/index_train_{split}.txt').astype(int)
    index_test = np.loadtxt(f'{dataset_dir}/index_test_{split}.txt').astype(int)

    # record unnormalized dataset (just to find the appropriate scaling)
    train_X = X_unnorm[index_train]
    train_y = y_unnorm[index_train]

    # compute normalization constants based on training set
    X_std = np.std(train_X, axis=0)
    X_std[X_std == 0] = 1.  # ensure we don't divide by zero
    X_mean = np.mean(train_X, axis=0)

    y_mean = np.mean(train_y)
    y_std = np.std(train_y)

    X_norm = (X_unnorm - X_mean) / X_std
    y_norm = (y_unnorm - y_mean) / y_std

    # record normalised dataset
    train_X = X_norm[index_train]
    test_X = X_norm[index_test]
    train_y = y_norm[index_train]
    test_y = y_norm[index_test]

    return train_X, train_y.flatten(), test_X, test_y.flatten(), y_mean, y_std
