import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import tqdm

from bbb.bnn import BNN, BNNLayer


class MyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)


def train_bbb(xs_train, ys_train, xs_test, hidden_dim=50, num_particles=20, num_epochs=20, batch_size=100, lr=1e-3):
    xs_train = torch.from_numpy(xs_train).type(torch.FloatTensor)
    ys_train = torch.from_numpy(ys_train.reshape((-1, 1))).type(torch.FloatTensor)
    xs_test = torch.from_numpy(xs_test).type(torch.FloatTensor)

    # dataset = MyDataset(xs_train, ys_train)
    # loader = DataLoader(dataset, batch_size=batch_size)

    bnn = BNN(BNNLayer(xs_train.shape[1], hidden_dim, activation='relu', prior_mean=0., prior_rho=0.),
              BNNLayer(hidden_dim, ys_train.shape[1], activation='none', prior_mean=0., prior_rho=0.))

    optim = torch.optim.Adam(bnn.parameters(), lr=lr)

    # Main training loop
    # pbar = tqdm.tqdm(total=num_epochs * (((len(dataset) - 1) // batch_size) + 1))
    t = time.time()
    for _ in tqdm.trange(num_epochs):
        # for xs_batch, ys_batch in loader:
        kl, lg_lklh = bnn.run_samples(x=xs_train, y=ys_train, n_samples=1, type='Gaussian')
        loss = BNN.loss_fn(kl, lg_lklh, 1)
        optim.zero_grad()
        loss.backward()
        optim.step()
        # pbar.update()
    t = time.time() - t

    pred_lst = [bnn.forward(xs_test, mode='MC').data.numpy().squeeze(1) for _ in range(num_particles)]

    pred = np.array(pred_lst).T
    pred_mean = pred.mean(axis=1)
    pred_std = pred.std(axis=1)
    return pred_mean, pred_std, t