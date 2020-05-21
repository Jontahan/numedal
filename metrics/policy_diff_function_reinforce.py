import numpy as np

from gw_collect import Gridworld
import pygame as pg
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from vicero.algorithms.reinforce import Reinforce

class LinRegNet(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(LinRegNet, self).__init__()
        self.linreg = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        x = torch.flatten(x)
        return F.softmax(self.linreg(x), dim=-1)
        
absolutely_all_diffs = []
def env_diff(env_a, env_b, iterations, step_size):
    all_states = env_a.get_all_states() + env_b.get_all_states()
        
    rf_a = Reinforce(env_a, LinRegNet(64, 4).double())
    rf_b = Reinforce(env_b, LinRegNet(64, 4).double())
    
    all_mean_diffs = []
    for ne in range(0, iterations):
        convergence_durations = []
        ql_agents = []
        
        rf_a.train(step_size)
        rf_b.train(step_size)
            
        env_diffs = []

        for state in all_states:
            state = torch.from_numpy(state)
            env_diffs.append(torch.sum((rf_a.policy_net(state) - rf_b.policy_net(state)) ** 2).item())

        print('{}/{} mean difference: {:.4f}'.format(ne + 1, iterations, np.mean(env_diffs)))
        all_mean_diffs.append(np.mean(env_diffs))
    
    absolutely_all_diffs.append(all_mean_diffs)
    return all_mean_diffs[-1]

envs = [
    (Gridworld(width=4, height=4, cell_size=32, agent_pos=(2, 0), food_pos=[(1, 3), (3, 3)]),
     Gridworld(width=4, height=4, cell_size=32, agent_pos=(0, 0), food_pos=[(1, 3), (3, 3)]))
]

for env_pair in envs:
    print(env_diff(env_pair[0], env_pair[1], 50, 100))

for diff in absolutely_all_diffs:
    plt.plot(diff)

#plt.savefig('test.png')
plt.show()