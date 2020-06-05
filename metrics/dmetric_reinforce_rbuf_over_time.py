#from .value_diff_function import *
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


import time

class LogRegNet(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(LogRegNet, self).__init__()
        self.linreg = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        x = torch.flatten(x)
        return F.softmax(self.linreg(x), dim=-1)

def get_diversity(env_list, training_iterations=1000, steps=10, verbose=False):
    
    policy_list = []
    
    for i in range(len(env_list)):
        policy = Reinforce(env_list[i], polinet=LogRegNet(64, 4).double())
        policy_list.append(policy)
    
    diversity_history = []
    
    for step in range(steps):
        print('step {}/{}'.format(step, steps))
        for i in range(len(env_list)):
            if verbose: print('Training value function for environment({}) {}/{}'.format(env_list[i].seed, i + 1, len(env_list)))
            start = time.time()
            policy_list[i].train(training_iterations)
            end = time.time()
            if verbose: print('Elapsed time: {:.2f}s'.format(end - start))
        
        diff_list = []
        for i in range(len(env_list)):
            for j in range(i + 1, len(env_list)):
                env_a = env_list[i]
                env_b = env_list[j]
                policy_a = policy_list[i]
                policy_b = policy_list[j]
                
                all_states = env_a.get_all_states() + env_b.get_all_states()
                #all_states = [sample[0].numpy() for sample in policy_a.memory] + [sample[0].numpy() for sample in policy_b.memory]
                env_diffs = []

                with torch.no_grad():
                    for state in all_states:
                        state = torch.from_numpy(state)
                        env_diffs.append(torch.sum((policy_a.policy_net(state) - policy_b.policy_net(state)) ** 2).item())

                diff_list.append(np.mean(env_diffs))
        diversity_history.append(np.mean(diff_list))

    return diversity_history