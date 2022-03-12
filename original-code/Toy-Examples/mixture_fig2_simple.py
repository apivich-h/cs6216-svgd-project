import numpy as np
import math 

np.random.seed(5432)

W1 = 1/3
W2 = 2/3
MU1 = -2
MU2 = 2
SIG1 = 1
SIG2 = 1

def MSE(Y, yh):
  return np.square(np.array(Y) - np.repeat(yh, Y.shape)).mean()

def mc_sample(sample_size):
  x0 = np.random.normal(-2,1,sample_size)
  x1 = np.random.normal(2,1,sample_size)
  ber = np.random.choice(2, sample_size, p=[1/3, 2/3])
  results = []
  for i in range(sample_size):
    result = x0[i] if ber[i] == 0 else x1[i]
    results.append(result)
  return np.array(results)

experiment_sample_sizes = [100, 200, 400, 800, 1600, 3200, 6400]

est_function_dict = {
  "E[X]": np.vectorize(lambda x: x),
  "E[X^2]": np.vectorize(lambda x: x^2),
  "E[cos(wx+b)]": lambda w, b : np.vectorize(lambda x: math.cos(w*x + b)) 
}

def cos_expectation(w, b):
  cos = math.cos
  if SIG1 == SIG2: return (math.e**(-(1/2) * SIG1**2 * w**2) (W1 * cos(b + MU1 * w)+W2 * cos(b + MU2 * w)))/(W1 + W2)
  else: raise Exception("True Expectation not calculated.")

est_true_value_dict = {
  "E[X]": lambda : (W1 * MU1 + W2 * MU2)/(W1 + W2),
  "E[X^2]": lambda : (MU1**2 * W1 + SIG1**2 * W1 + MU2**2 * W2 + SIG2**2 * W2)/(W1 + W2),
  "E[cos(wx+b)]": cos_expectation 
}

sample_function_dict = {
  "MC": mc_sample
}

EXPERIMENT_REPEAT = 20

for est_func_name in est_function_dict:
  est_function = est_function_dict[est_func_name]
  est_true_function = est_true_value_dict[est_func_name]
  
  for sample_type in sample_function_dict:
    sample_func = sample_function_dict[sample_type]

    results_list = []
    for sample_size in experiment_sample_sizes:
      repeat_experiment_list = []
      for i in range(EXPERIMENT_REPEAT):
        est_true_val = None
        if est_func_name == "E[cos(wx+b)]":
          w = np.random.normal(0, 1)
          b = np.random.uniform(0,2*math.pi)
          est_function = est_function(w, b)
          est_true_val = est_true_function(w, b)
        else:
          est_true_val = est_true_function()
        
        sample = sample_func(sample_size)
        estimation = est_function(sample)
        repeat_experiment_list.append(estimation)
      assert len(repeat_experiment_list) == EXPERIMENT_REPEAT
      emp_mean = np.mean(repeat_experiment_list)
      true_mean = 
      mse = MSE()
      results_list.append([sample_size, repeat_experiment_list])

