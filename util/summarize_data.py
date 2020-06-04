import numpy as np
import matplotlib.pyplot as plt

def view_data(path, xlabel='', ylabel='', x_end=None, show=True, color='g'):
    with open(path) as f:
        lines = [float(line) for line in f.readlines()]
        print(lines[-1])
        return lines[-1]

base = 9
lst = []
for i in range(5):
    lst.append(view_data('out/identity_test_{}_all_states/data_{}.txt'.format(base, i)))
print(np.mean(lst))