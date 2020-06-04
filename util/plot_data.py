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


def plot_data(data, name, x=None, xlabel='', ylabel='', x_end=None, show=True, color='g'):
    if x is None:
        x = list(range(0, x_end if x_end is not None else len(data), x_end // len(data) if x_end is not None else 1))
    plt.plot(x, data, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig('{}_plot.png'.format(name))
    if show: plt.show()

plt.rcParams.update({'font.size': 12})
plot_data([723, 2067, 4081, 6714, 9290, 11028], 'set_scale_time',x=[5,10,15,20,25,30], xlabel='Set Size', ylabel='Duration (s)', color='b')