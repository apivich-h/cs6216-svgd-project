
# %%
import numpy as np
import math 
from mixture1d import Mixture1d
from svgd import SVGD

W1 = 1/3
W2 = 2/3
MU1 = -2
MU2 = 2
SIG1 = 1
SIG2 = 1

EXPERIMENT_REPEAT = 100
COS_PARAM_REPEAT = 20

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
sample_cache = {}

def add_sample_in_cache(method, sample_size, sample):
  global sample_cache
  if method not in sample_cache: sample_cache[method] = {}
  method_sample_dict = sample_cache[method] 
  if sample_size not in method_sample_dict:
    method_sample_dict[sample_size] = []
  print(f"# add_sample_in_cache  method={method} sample_size={sample_size} sample_idx={len(method_sample_dict[sample_size])}")
  method_sample_dict[sample_size].append(sample)

def repeat_sample_add(sample_method,  sample_size, times):
    sample_func = sample_function_dict[sample_method]
    for i in range(times): add_sample_in_cache(sample_method, sample_size, sample_func(sample_size))

def dump_cache(method, sample_size):
  print("----- dump sample cache: ", method, sample_size)
  fname = f"./output/sample_cache_{method}_{sample_size}.csv"
  np.savetxt(fname, sample_cache[method][sample_size])

np.random.seed(5433)
for sample_size in experiment_sample_sizes:
  repeat_sample_add("MC", sample_size, 100 * 20)
  dump_cache("MC", sample_size)
  repeat_sample_add("SVGD", sample_size, 100 * 20)
  dump_cache("SVGD", sample_size)


# %%

np.random.seed(5432)
def next_sample_getter_gen(method):
  idx = 0
  def _next_sample_getter(sample_size):
    nonlocal idx
    result = sample_cache[method][sample_size][idx]
    idx += 1
    return result
  return _next_sample_getter

all_result_dict = {}

def repeat_exp_get_mse(next_sample_getter, sample_size, using_est_function, expected_mean):
  repeat_experiment_emp_list = []
  repeat_experiment_expected_list = [expected_mean for i in range(EXPERIMENT_REPEAT)]
  for i in range(EXPERIMENT_REPEAT):
    sample = next_sample_getter(sample_size)
    estimation = np.mean(using_est_function(sample))
    repeat_experiment_emp_list.append(estimation)
  assert len(repeat_experiment_emp_list) == EXPERIMENT_REPEAT and len(repeat_experiment_expected_list) == EXPERIMENT_REPEAT
  mse = MSE(repeat_experiment_emp_list, repeat_experiment_expected_list)
  return mse

print ("=============== get sample & calculate & plot ===============")
for est_func_name in est_function_dict:
  est_function = est_function_dict[est_func_name]
  est_true_function = est_true_value_dict[est_func_name]
  all_result_dict[est_func_name] = {}

  for sample_type in sample_function_dict:
    sample_getter = next_sample_getter_gen(sample_type)
    all_result_dict[est_func_name][sample_type] = []
    results_list = all_result_dict[est_func_name][sample_type]

    for sample_size in experiment_sample_sizes:
      repeat_experiment_emp_list = []
      repeat_experiment_expected_list = []

      if est_func_name == "E[cos(wx+b)]":
        mse_list = []
        for j in range(COS_PARAM_REPEAT):
          w = np.random.normal(0, 1)
          b = np.random.uniform(0, 2 * math.pi)
          using_est_function = est_function(w, b)
          est_true_val = est_true_function(w, b)
          mse = repeat_exp_get_mse(sample_getter, sample_size, using_est_function, est_true_val)
          print(f"calculate estimator (PARAM idx={j} w={w}, b={b}) MSE:", mse, " sample_size:", sample_size, " sample_type:", sample_type, " est_func_name:", est_func_name)
          mse_list.append(mse)
        mean_mse = np.mean(mse_list)
        print(f"calculate estimator (PARAM MEAN) MSE:", mean_mse, " sample_size:", sample_size, " sample_type:", sample_type, " est_func_name:", est_func_name)
        

      else:
        using_est_function = est_function
        est_true_val = est_true_function()
        mse = repeat_exp_get_mse(sample_getter, sample_size, using_est_function, est_true_val)
        print("calculate estimator (ONCE) MSE:", mse, " sample_size:", sample_size, " sample_type:", sample_type, " est_func_name:", est_func_name)
        results_list.append([sample_size, mse])

# %%
import json
with open("./output/mixture_fig2.json", 'r') as f:
  all_result_dict = json.load(f)

import matplotlib.pyplot as plt
fig_keys = list(est_true_value_dict.keys())
num_plots = len(fig_keys)
fig = plt.figure(figsize=(4 * num_plots, 4))
axs = fig.subplots(nrows=1, ncols=num_plots, sharex=True, sharey=False)
for i, key in enumerate(fig_keys):
  estimate_dict = all_result_dict[key]
  ax = axs[i]
  ax.set_title(f'Estimating {key}')
  ax.set_yscale('log')
  for method_key in estimate_dict:
    estimate_mse_data = estimate_dict[method_key]
    X, Y = np.array(estimate_mse_data).T
    ax.plot(X, Y)

fig.savefig('./output/figure2.png')
# %%
