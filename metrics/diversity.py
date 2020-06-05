import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from vicero.algorithms.reinforce import Reinforce
from vicero.algorithms.deepqlearning import DQN

class LogRegNet(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(LogRegNet, self).__init__()
        self.linreg = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        x = torch.flatten(x)
        return F.softmax(self.linreg(x), dim=-1)

class LinRegNet(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(LinRegNet, self).__init__()
        self.linreg = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        x = torch.flatten(x)
        return self.linreg(x)


# learning_alg = 'dqn'/'reinforce'
# state_dist = 'memory'/'full'

def get_diversity(env_list, learning_alg, state_dist, softmax=False, training_iterations=1000, steps=10, verbose=False, input_size=64):
    assert not (learning_alg == 'reinforce' and softmax)
    expert_agents = []
    
    for i in range(len(env_list)):
        if learning_alg == 'reinforce':
            agent = Reinforce(env_list[i], polinet=LogRegNet(input_size, 4).double())
        elif learning_alg == 'dqn':
            agent = DQN(env_list[i], qnet=LinRegNet(input_size, 4).double(), plotter=None, render=False, memory_length=2000, gamma=.99, alpha=.001, epsilon_start=0.3)
      
        expert_agents.append(agent)
    
    diversity_history = []
    
    for step in range(steps):
        print('step {}/{}'.format(step, steps))
        for i in range(len(env_list)):
            if verbose: print('Training expert agent for environment({}) {}/{}'.format(env_list[i].seed, i + 1, len(env_list)))
            start = time.time()
            
            if learning_alg == 'reinforce':
                expert_agents[i].train(training_iterations)
            elif learning_alg == 'dqn':
                expert_agents[i].train(training_iterations, 4)
            
            end = time.time()
            if verbose: print('Elapsed time: {:.2f}s'.format(end - start))
        
        diff_list = []
        for i in range(len(env_list)):
            for j in range(i + 1, len(env_list)):
                env_a = env_list[i]
                env_b = env_list[j]
                agent_a = expert_agents[i]
                agent_b = expert_agents[j]
                
                if state_dist == 'full':
                    all_states = env_a.get_all_states() + env_b.get_all_states()
                elif state_dist == 'memory':
                    all_states = [sample[0].numpy() for sample in agent_a.memory] + [sample[0].numpy() for sample in agent_b.memory]
                env_diffs = []

                with torch.no_grad():
                    for state in all_states:
                        state = torch.from_numpy(state)
                        
                        if learning_alg == 'reinforce':
                            env_diffs.append(torch.sum((agent_a.policy_net(state) - agent_b.policy_net(state)) ** 2).item())
                        elif learning_alg == 'dqn':
                            if softmax:
                                env_diffs.append(torch.sum((F.softmax(agent_a.qnet(state), dim=-1) - F.softmax(agent_b.qnet(state), dim=-1)) ** 2).item())
                            else:
                                env_diffs.append(torch.sum((agent_a.qnet(state) - agent_b.qnet(state)) ** 2).item())

                diff_list.append(np.mean(env_diffs))
        diversity_history.append(np.mean(diff_list))

    return diversity_history