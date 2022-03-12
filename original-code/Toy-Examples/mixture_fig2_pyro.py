# !jupyter nbextension enable --py widgetsnbextension

# %%
import os
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist

smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.8.0')

print(pyro.__version__)

def model(data):
    m = pyro.sample("m", dist.Normal(0, 1))
    sd = pyro.sample("sd", dist.LogNormal(m, 1))
    with pyro.plate("N", len(data)):
        pyro.sample("obs", dist.Normal(m, sd), obs=data)

# %%
data = torch.ones(10)
pyro.render_model(model, model_args=(data,))
# %%
