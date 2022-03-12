## Original Code Reference `Stein-Variational-Gradient-Descent`

Copied from

https://github.com/DartML/Stein-Variational-Gradient-Descent

### Installation 
This code won't work on python 3.9+ due to theano. Using conda to create python3.7 environment.
```
conda create --name cs6216proj-original
conda activate cs6216proj-original
conda install python=3.7

```

### Run 

To run `bayesian_logistic_regression.py` and `multivariate_normal.py`:

```
pip install sklearn
python3 bayesian_logistic_regression.py
python3 multivariate_normal.py
```

To run `bayesian_nn.py`:

```
pip install theano==0.8.2
python3 bayesian_nn.py
```

## Simple code using the original SVGD impl `Toy-Examples`

Same dependencies and conda environment as previous.

```
python3 mixture1d.py
python3 mixture1d_harder.py
```
Then use the content of `mixture1d.csv` for visualization.

## Simple code for figure 2

Same dependencies and conda environment as previous.

```
python3 mixure_fig2_simple.py
```

## Try to use Pyro (not important for figure 2)
To run `mixture_fig2.py` (on RTX3090 GPU), install the following stuff (assuming conda environment is `cs6231proj`):
```
conda activate cs6231proj
sudo apt install graphviz
conda install pytorch torchvision torchaudio cudatoolkit=11.3
conda install ipykernel
pip install ipywidgets widgetsnbextension pandas-profiling
pip install pyro-ppl
pip install graphviz
```