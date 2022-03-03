import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm
import pandas as pd

import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.svgd import SVGD, RBFSteinKernel

"""
based on some code from https://www.kaggle.com/carlossouza/simple-bayesian-neural-network-in-pyro
and some samples at PyroModule https://docs.pyro.ai/en/stable/nn.html
"""


def generate_data():
    x = np.linspace(0, 0.5, 1000)
    eps = 0.02 * np.random.randn(x.shape[0])
    y = x - 0.2 * x ** 2 + 0.3 * np.sin(2 * np.pi * (x + eps)) + 0.3 * np.sin(4 * np.pi * (x + eps)) + eps
    return x, y


# class Model(PyroModule):
#
#     def __init__(self, input_dim=1, hidden_dim=50, output_dim=1, activation=nn.ReLU):
#         super().__init__()
#
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.activation = activation
#
#         self.fcs = PyroModule[nn.Sequential](
#             PyroModule[nn.Linear](self.input_dim, self.hidden_dim),
#             PyroModule[self.activation](),
#             PyroModule[nn.Linear](self.hidden_dim, self.output_dim),
#         )
#
#         # sample the weight of the NN
#         self.fcs[0].weight = PyroSample(dist.Normal(0., 1.).expand([self.hidden_dim, self.input_dim]).to_event(2))
#         self.fcs[0].bias = PyroSample(dist.Normal(0., 1.).expand([self.hidden_dim]).to_event(1))
#         self.fcs[2].weight = PyroSample(dist.Normal(0., 1.).expand([self.output_dim, self.hidden_dim]).to_event(2))
#         self.fcs[2].bias = PyroSample(dist.Normal(0., 1.).expand([self.output_dim]).to_event(1))
#
#     def forward(self, x, y=None):
#         x = x.reshape(-1, 1)
#         mu = self.fcs(x).squeeze()
#         sigma = pyro.sample("sigma", dist.Uniform(0., 1.))
#         with pyro.plate("data", x.shape[0]):
#             obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
#             return obs


def model(xs, ys):
    xs = xs.reshape(-1, 1)
    D1 = torch.Size([1, 50])
    w1 = pyro.sample('w1', dist.Normal(loc=torch.zeros(D1), scale=torch.ones(D1)))
    b1 = pyro.sample('b1', dist.Normal(loc=torch.zeros(D1[1]), scale=torch.ones(D1[1])))
    D2 = torch.Size([50, 1])
    w2 = pyro.sample('w2', dist.Normal(loc=torch.zeros(D2), scale=torch.ones(D2)))
    b2 = pyro.sample('b2', dist.Normal(loc=torch.zeros(D2[1]), scale=torch.ones(D2[1])))
    sigma = pyro.sample('sigma', dist.Uniform(0, 1))
    with pyro.plate("map", len(xs), dim=-2):
        print(xs.shape, w1.shape, (w1 * xs).shape, b1.shape)
        h1 = nn.functional.relu(torch.matmul(xs, w1) + b1)
        print(h1.shape, w2.shape, b2.shape)
        mu = torch.matmul(h1, w2)
        return pyro.sample("obs", dist.Normal(mu, sigma), obs=ys)


if __name__ == '__main__':

    xs, ys = generate_data()
    x_train = torch.from_numpy(xs).float()
    y_train = torch.from_numpy(ys).float()

    # model = Model()

    pyro.clear_param_store()
    guide = AutoDiagonalNormal(model)
    adam = pyro.optim.Adam({"lr": 1e-3})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    bar = trange(20000)
    for epoch in bar:
        loss = svi.step(x_train, y_train)
        bar.set_postfix(loss=f'{loss / xs.shape[0]:.3f}')

    predictive = Predictive(model, guide=guide, num_samples=500)
    x_test = torch.linspace(-0.5, 1, 3000)
    preds = predictive(x_test)

    y_pred = preds['obs'].T.detach().numpy().mean(axis=1)
    y_std = preds['obs'].T.detach().numpy().std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, ys, 'o', markersize=1)
    ax.plot(x_test, y_pred)
    ax.fill_between(x_test, y_pred - y_std, y_pred + y_std,
                    alpha=0.5, color='#ffcd3c')

    fig.savefig('figs/bayesian_nn_svi')

    # pyro.clear_param_store()
    # kernel = RBFSteinKernel()
    # adam = pyro.optim.Adam({"lr": 0.1})
    # svgd = SVGD(model, kernel, adam, num_particles=100, max_plate_nesting=0)
    #
    # for i in trange(50):
    #     svgd.step(x_train, y_train)

# import numpy as np
# import matplotlib.pyplot as plt
# import tqdm
#
# import torch
# import pyro
# import pyro.distributions as dist
# from pyro.optim import Adam
# from pyro.contrib.bnn import HiddenLayer
# from pyro.distributions import constraints
#
# from pyro.infer.svgd import SVGD, RBFSteinKernel
#
#
# class BNN(torch.nn.Module):
#
#     def __init__(self, input_dim=1, n_hidden=50, output_dim=1):
#         super(BNN, self).__init__()
#         self.input_dim = input_dim
#         self.n_hidden = n_hidden
#         self.output_dim = output_dim
#
#     def model(self, input, output=None):
#         input = input.view(-1, self.input_dim)
#         n_input = input.size(0)
#
#         # first layer
#         D1 = (self.input_dim, self.n_hidden)
#         mean_1 = pyro.sample('mean1', dist.Normal(torch.zeros(D1), torch.ones(D1)))
#         scale_1 = pyro.sample('scale_1', dist.Normal(torch.zeros(D1), torch.ones(D1)))
#
#         # output layer
#         D2 = (self.n_hidden, self.output_dim)
#         mean_2 = pyro.sample('mean2', dist.Normal(torch.zeros(D2), torch.ones(D2)))
#         scale_2 = pyro.sample('bias2', dist.Normal(torch.zeros(D2), torch.ones(D2)))
#
#         with pyro.plate('data', size=n_input):
#             # Sample first hidden layer
#             print(input.shape, mean_1.shape, scale_1.shape)
#             hidden = pyro.sample('hidden1', HiddenLayer(input, mean_1, scale_1,
#                                                         non_linearity=torch.nn.functional.relu))
#             model_out = pyro.sample('hidden1', HiddenLayer(hidden, mean_2, scale_2,
#                                                            non_linearity=lambda x: x))
#             return model_out
#
#     # def guide(self, input, output=None):
#     #     input = input.view(-1, self.input_dim)
#     #     n_input = input.size(0)
#     #
#     #     # first layer
#     #     mean_1 = pyro.param('mean1', torch.randn(self.input_dim, self.n_hidden))
#     #     scale_1 = pyro.param('scale1', torch.ones(self.input_dim, self.n_hidden),
#     #                          constraint=constraints.greater_than(1e-4))
#     #
#     #     # output layer
#     #     mean_2 = pyro.param('mean2', torch.randn(self.n_hidden, self.output_dim))
#     #     scale_2 = pyro.param('scale2', torch.ones(self.n_hidden, self.output_dim),
#     #                          constraint=constraints.greater_than(1e-4))
#     #
#     #     with pyro.plate('data', size=n_input):
#     #         # Sample first hidden layer
#     #         hidden = pyro.sample('h1', HiddenLayer(input, mean_1, scale_1,
#     #                                                non_linearity=torch.nn.functional.relu))
#     #         model_out = pyro.sample('h1', HiddenLayer(hidden, mean_2, scale_2,
#     #                                                   non_linearity=lambda x: x))
#
#     def forward(self, input):
#         input = input.view(-1, self.input_dim)
#         n_input = input.size(0)
#         with pyro.plate('data', size=n_input):
#             t = pyro.poutine.trace(self.guide).get_trace(input)
#             return t
#
#
# if __name__ == '__main__':
#     xs = np.linspace(-10., 10.)
#     ys = np.sin(xs) + np.cos(2 * xs - 3.)
#
#     xs_tens = torch.Tensor(xs)
#     ys_tens = torch.Tensor(ys)
#
#     pyro.clear_param_store()
#     print("SVGD")
#
#     bnn = BNN(input_dim=1, n_hidden=50, output_dim=1)
#
#     kernel = RBFSteinKernel()
#     adam = Adam({"lr": 0.1})
#     svgd = SVGD(bnn.model, kernel, adam, num_particles=100, max_plate_nesting=2)
#
#     for i in tqdm.trange(50):
#         svgd.step(xs_tens, ys_tens)
#
#     # model = pyrhandlers.substitute(handlers.seed(model, rng_key), samples)
#     # # note that Y will be sampled in the model because we pass Y=None here
#     # model_trace = handlers.trace(model).get_trace(X=X, Y=None, D_H=D_H)
#     # return model_trace["Y"]["value"]
#
#     # optim = pyroopt.SGD({'lr': lr, 'momentum': momentum, 'nesterov': True})
#     # elbo = TraceMeanField_ELBO()
#     # svi = SVI(self.model, self.guide, optim, elbo)
#     # kl_factor = loader.batch_size / len(loader.dataset)
#     # for i in range(num_epochs):
#     #     total_loss = 0.0
#     #     total = 0.0
#     #     correct = 0.0
#     #     for images, labels in loader:
#     #         loss = svi.step(images.cuda(), labels.cuda(), kl_factor=kl_factor)
#     #         pred = self.forward(images.cuda(), n_samples=1).mean(0)
#     #         total_loss += loss / len(loader.dataset)
#     #         total += labels.size(0)
#     #         correct += (pred.argmax(-1) == labels.cuda()).sum().item()
#     #         param_store = pyro.get_param_store()
#     #     print(f"[Epoch {i + 1}] loss: {total_loss:.5E} accuracy: {correct / total * 100:.5f}")
