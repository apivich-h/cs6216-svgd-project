
# %%
import numpy as np
import math 
from mixture1d import Mixture1d
from svgd import SVGD

np.random.seed(5432)

W1 = 1/3
W2 = 2/3
MU1 = -2
MU2 = 2
SIG1 = 1
SIG2 = 1

EXPERIMENT_REPEAT = 20

def MSE(Y, TY):
  return np.square(np.array(Y) - np.array(TY)).mean()

experiment_sample_sizes = [round(10* (1.49534878**i)) for i in range(9)]
# experiment_sample_sizes = [10, 15, 22, 33, 50, 75, 112, 167, 250]
# experiment_sample_sizes = [100, 200, 400, 800, 1600, 3200, 6400]
print("experiment_sample_sizes:", experiment_sample_sizes)

est_function_dict = {
  "E[X]": np.vectorize(lambda x: x),
  "E[X^2]": np.vectorize(lambda x: x**2),
  "E[cos(wx+b)]": lambda w, b : np.vectorize(lambda x: math.cos(w*x + b)) 
}

def cos_expectation(w, b):
  cos = math.cos
  if SIG1 == SIG2: return (math.e**(-(1/2) * SIG1**2 * w**2) * (W1 * cos(b + MU1 * w)+W2 * cos(b + MU2 * w)))/(W1 + W2)
  else: raise Exception("True Expectation not calculated.")

est_true_value_dict = {
  "E[X]": lambda : (W1 * MU1 + W2 * MU2)/(W1 + W2),
  "E[X^2]": lambda : (MU1**2 * W1 + SIG1**2 * W1 + MU2**2 * W2 + SIG2**2 * W2)/(W1 + W2),
  "E[cos(wx+b)]": cos_expectation 
}

def mc_sample(sample_size):
  x0 = np.random.normal(-2,1,sample_size)
  x1 = np.random.normal(2,1,sample_size)
  ber = np.random.choice(2, sample_size, p=[1/3, 2/3])
  results = []
  for i in range(sample_size):
    result = x0[i] if ber[i] == 0 else x1[i]
    results.append(result)
  return np.array(results)

def svgd_sample(sample_size):
  model = Mixture1d(W1, MU1, SIG1, W2, MU2, SIG2)
  x0 = np.random.normal(-10,1,[sample_size,1])
  x_after = SVGD().update(x0, model.dlnprob, n_iter=500, stepsize=0.25)
  return x_after.flatten()

sample_function_dict = {
  "MC": mc_sample,
  "SVGD": svgd_sample
}

# %%
all_result_dict = {}
for est_func_name in est_function_dict:
  est_function = est_function_dict[est_func_name]
  est_true_function = est_true_value_dict[est_func_name]
  all_result_dict[est_func_name] = {}

  for sample_type in sample_function_dict:
    sample_func = sample_function_dict[sample_type]
    all_result_dict[est_func_name][sample_type] = []
    results_list = all_result_dict[est_func_name][sample_type]
    for sample_size in experiment_sample_sizes:
      repeat_experiment_emp_list = []
      repeat_experiment_expected_list = []
      for i in range(EXPERIMENT_REPEAT):
        est_true_val = None
        if est_func_name == "E[cos(wx+b)]":
          w = np.random.normal(0, 1)
          b = np.random.uniform(0, 2 * math.pi)
          using_est_function = est_function(w, b)
          est_true_val = est_true_function(w, b)
        else:
          using_est_function = est_function
          est_true_val = est_true_function()
        sample = sample_func(sample_size)
        estimation = np.mean(using_est_function(sample))
        repeat_experiment_emp_list.append(estimation)
        repeat_experiment_expected_list.append(est_true_val)

      assert len(repeat_experiment_emp_list) == EXPERIMENT_REPEAT \
        and len(repeat_experiment_expected_list) == EXPERIMENT_REPEAT

      mse = MSE(repeat_experiment_emp_list, repeat_experiment_expected_list)
      print("calculate estimator MSE:", mse, " sample_size:", sample_size, " sample_type:", sample_type, " est_func_name:", est_func_name)
      results_list.append([sample_size, mse])

# %%
import json
with open("./output/mixture_fig2.json", 'w') as f:
  f.write(json.dumps(all_result_dict, indent=2))

# %%
import matplotlib

# %%
