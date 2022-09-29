import re
import os
import numpy as np

results_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                "plots/testing_results.txt")
my_open = open(results_file_path, "r")

data = []
data_for_each_method = []
lines = my_open.readlines()
t = 0

for line in lines:
    if line.split():
        if 'iCR' in line and t!=0:
            t = 0
            float_data = np.array(data_for_each_method).astype(float)
            reward_mean = np.round(np.mean(float_data[:, 0]), 2)
            reward_std = np.round(np.std(float_data[:, 0]), 2)
            error_mean = np.round(np.mean(float_data[:, 1]), 2)

            data.append([reward_mean, reward_std, error_mean])
            data_for_each_method = []
        else:
            if 'iCR' not in line:
                line = line.strip("\n").split()
                data_for_each_method.append(line)
                t += 1

float_data = np.array(data_for_each_method).astype(float)
reward_mean = np.round(np.mean(float_data[:, 0]), 2)
reward_std = np.round(np.std(float_data[:, 0]), 2)
error_mean = np.round(np.mean(float_data[:, 1]), 2)
data.append([reward_mean, reward_std, error_mean])
data_for_each_method = []

print(data)


