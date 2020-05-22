import numpy as np
import matplotlib.pyplot as plt

def plot_data_from_file(path, xlabel='', ylabel='', x_end=None, show=True, color='g'):
    with open(path) as f:
        lines = [float(line) for line in f.readlines()]
        x = list(range(0, x_end if x_end is not None else len(lines), x_end // len(lines) if x_end is not None else 1))
        plt.plot(x, lines, color=color)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig('{}_plot.png'.format(path.split('.')[0]))
        if show: plt.show()