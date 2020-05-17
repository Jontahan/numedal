import numpy as np

from gw_collect import Gridworld
import pygame as pg
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from vicero.algorithms.qlearning import Qlearning

#env = Gridworld(width=6, height=6, cell_size=32, agent_pos=(0, 3), food_pos=[(0, 0), (3, 3), (4, 5), (2, 0)])
env_a = Gridworld(width=4, height=4, cell_size=32, agent_pos=(0, 0), food_pos=[(0, 3), (3, 3)])
env_b = Gridworld(width=4, height=4, cell_size=32, agent_pos=(0, 0), food_pos=[(0, 3), (3, 3)])

pg.init()
screen = pg.display.set_mode((env_a.cell_size * env_a.width, env_a.cell_size * env_a.height))
env_a.screen = screen
clock = pg.time.Clock()

def plot(history):
        plt.figure(2)
        plt.clf()
        durations_t = torch.FloatTensor(history)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy(), c='lightgray', linewidth=1)

        his = 50
        if len(durations_t) >= his:
            means = durations_t.unfold(0, his, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(his - 1), means))
            plt.plot(means.numpy(), c='green')
            
        plt.pause(0.001)

gamma = .95
alpha = .002

all_mean_diffs = []

all_states = env_a.get_all_states()        
ql_a = Qlearning(env_a, n_states=len(all_states), n_actions=env_a.action_space.n, plotter=plot, epsilon=1.0, epsilon_decay=lambda e, i: e * .998)
ql_b = Qlearning(env_b, n_states=len(all_states), n_actions=env_b.action_space.n, plotter=plot, epsilon=1.0, epsilon_decay=lambda e, i: e * .998)
        
for ne in range(0, 20):
    #np.random.seed(10)
    num_episodes = 100 #* ne
    convergence_durations = []
    ql_agents = []
    
    ql_a.train(num_episodes)
    ql_b.train(num_episodes)
        
    env_diffs = []
    total_visits = []
    for i in range(len(ql_a.Q)):
        total_visits.append(ql_a.state_visits[i] + ql_b.state_visits[i])

    normalized_visits = total_visits / np.linalg.norm(total_visits)

    for i in range(len(ql_a.Q)):
        for a in range(env_a.action_space.n):
            env_diffs.append(normalized_visits[i] * ((ql_a.Q[i][a] - ql_b.Q[i][a]) ** 2))

    print('mean difference: {}'.format(np.mean(env_diffs)))
    all_mean_diffs.append(np.mean(env_diffs))

plt.close()
plt.plot(all_mean_diffs)
plt.show()
#print('mean duration: {}'.format(np.mean(convergence_durations)))

# std < 1 for this config