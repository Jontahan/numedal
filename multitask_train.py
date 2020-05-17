import numpy as np
import matplotlib.pyplot as plt
import torch

from vicero.algorithms.deepqlearning import DQN
from gw_collect import Gridworld
from multitask_env import MultitaskEnvironment
from common.nn import LinRegNet

def plot(history):
    plt.figure(2)
    plt.clf()
    durations_t = torch.DoubleTensor(history)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy(), c='lightgray', linewidth=1)

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), c='green')
            
    plt.pause(0.001)

env_list = []

for i in [8, 0, 2, 4, 7]:
    env_list.append(Gridworld(width=4, height=4, cell_size=32, seed=i))

env = MultitaskEnvironment(env_list)

dqn = DQN(env, qnet=LinRegNet(64, 4).double(), plotter=plot, render=False, memory_length=2000, gamma=.99, alpha=.001, epsilon_start=0.1, plot_durations=True)
dqn.train(10000, 4, plot=True)

for seed in env.env_scores:
    print('seed={}, score={}'.format(seed, np.mean(env.env_scores[seed])))