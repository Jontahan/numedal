import numpy as np

from gw_collect import Gridworld
import pygame as pg
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from vicero.algorithms.deepqlearning import DQN

class LinRegNet(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(LinRegNet, self).__init__()
        self.linreg = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        x = torch.flatten(x)
        return self.linreg(x)

absolutely_all_diffs = []
def env_diff(env_a, env_b, iterations, step_size):
    all_states = env_a.get_all_states() + env_b.get_all_states()
        
    dqn_a = DQN(env_a, qnet=LinRegNet(64, 4).double(), plotter=None, render=False, memory_length=2000, gamma=.99, alpha=.001, epsilon_start=0.1)
    dqn_b = DQN(env_b, qnet=LinRegNet(64, 4).double(), plotter=None, render=False, memory_length=2000, gamma=.99, alpha=.001, epsilon_start=0.1)

    all_mean_diffs = []
    for ne in range(0, iterations):
        convergence_durations = []
        ql_agents = []
        
        dqn_a.train(step_size, 4, plot=False)
        dqn_b.train(step_size, 4, plot=False)
            
        env_diffs = []

        for state in all_states:
            state = torch.from_numpy(state)
            env_diffs.append(torch.sum((dqn_a.qnet(state) - dqn_b.qnet(state)) ** 2).item())

        print('{}/{} mean difference: {:.4f}'.format(ne + 1, iterations, np.mean(env_diffs)))
        all_mean_diffs.append(np.mean(env_diffs))
    
    absolutely_all_diffs.append(all_mean_diffs)
    return all_mean_diffs[-1]


envs = [
    (Gridworld(width=4, height=4, cell_size=32, agent_pos=(2, 0), food_pos=[(0, 3), (3, 3)]),
     Gridworld(width=4, height=4, cell_size=32, agent_pos=(2, 0), food_pos=[(1, 3), (3, 3)]))
]

for env_pair in envs:
    print(env_diff(env_pair[0], env_pair[1], 10, 10))

for diff in absolutely_all_diffs:
    plt.plot(diff)
#plt.savefig('test.png')
#plt.show()