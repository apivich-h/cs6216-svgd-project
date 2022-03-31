# SVGD

Paper: `https://arxiv.org/pdf/1608.04471.pdf`

All the code files mentioned below are written by us or partially written by us (details are in the code comments and the `Links to public codebases` session below).   
The original paper's code is copied into this folder for a reference: [original-code/Stein-Variational-Gradient-Decent](./original-code/Stein-Variational-Gradient-Descent/)  

## Toy Example (Gaussian mixture 1d)

- In browser demo: [svgd_toy.html](https://apivich-h.github.io/cs6216-svgd-project/svgd_toy.html)

- For figure 1
  - toy/Toy-Examples/mixture1d.py
  - toy/Toy-Examples/mixture1d_numpyro.py

- For figure 2
  - toy/Toy-Examples/mixture_fig2_simple.py

## Running time vs number of particles

- For figure 3
  - toy/Toy-Examples/simple_scalability_test.py
  - toy/Toy-Examples/simple_scalability_test_numpyro.py

## Small scale Logistic Regression

- bayesian_logistic_realdata.py

## Bayesian Logistic Regression on binary Covertype
- For figure 5
  - bayeisan_logistic_covtype.py
  - dsvi_logistic_covtype.py

## Bayesian Neural network

- bayesian_nn_uci.py

## Links to public codebases used in this project:
### SVGD:
- https://github.com/dilinwang820/Stein-Variational-Gradient-Descent (the original implementation of SVGD from the authors of the paper)
- https://num.pyro.ai/en/latest/examples/stein_bnn.html (example of the SVGD module on NumPyro)

### Other Algorithms Used
- https://github.com/wiseodd/MCMC/blob/master/algo/sgld.py (SGLD algorithm, used as comparison for large-scale logistic regression tasks)
- https://github.com/matsur8/dsvi (DSVI algorithm)
- https://github.com/HIPS/Probabilistic-Backpropagation (Probabilistic Back-propagation algorithm, used as comparison in BNN training)

### Dataset Used
- https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html (data for the small logistic regression task)
- https://github.com/ratschlab/bnn_priors/tree/main/bnn_priors (data for UCI dataset, with splitting of data into train-test sets)
- https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/blob/master/data/covertype.mat (binary Covertype dataset used in the original paper)
