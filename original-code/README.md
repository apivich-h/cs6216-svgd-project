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
```
Then use the content of `mixture1d.csv` for visualization.