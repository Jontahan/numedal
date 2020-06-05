import numpy as np
import matplotlib.pyplot as plt
import torch

from scipy.signal import savgol_filter
from vicero.algorithms.deepqlearning import DQN
from gw_collect import Gridworld
from multitask_env import MultitaskEnvironment
from common.nn import LinRegNet, NeuralNet

training_iterations = 500
repetitions = 10

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

print('Set A')
histories_a = []

env_list = []


for i in range(10):
    env_list.append(Gridworld(width=4, height=4, cell_size=32, seed=(20 + i)))
#for i in [8, 0, 2, 4, 7]:
#    env_list.append(Gridworld(width=4, height=4, cell_size=32, seed=i))

env = MultitaskEnvironment(env_list)

for _ in range(repetitions):
    print('x')
    dqn = DQN(env, qnet=NeuralNet(64, 4).double(), plotter=plot, render=False, memory_length=2000, gamma=.99, alpha=.001, epsilon_start=0.1, plot_durations=True)
    dqn.train(training_iterations, 4, plot=False)
    histories_a.append(dqn.history)

#for seed in env.env_scores:
#    print('seed={}, score={}'.format(seed, np.mean(env.env_scores[seed])))

print('Set B')
histories_b = []

env_list = []

for i in [22, 23, 24, 25, 51]:
    env_list.append(Gridworld(width=4, height=4, cell_size=32, seed=i))

env = MultitaskEnvironment(env_list)

for _ in range(repetitions):
    print('x')
    dqn = DQN(env, qnet=LinRegNet(64, 4).double(), plotter=plot, render=False, memory_length=2000, gamma=.99, alpha=.001, epsilon_start=0.1, plot_durations=True)
    dqn.train(training_iterations, 4, plot=False)
    histories_b.append(dqn.history)

smooth_factor = 100
plt.close()
for history in histories_a:
    mean_history = []
    for i in range(1, len(history)):
        if i < smooth_factor:
            mean_history.append(np.mean(history[:i]))
        else:
            mean_history.append(np.mean(history[i - smooth_factor : i]))
    plt.plot(mean_history)
plt.show()

plt.close()
for history in histories_b:
    mean_history = []
    for i in range(1, len(history)):
        if i < smooth_factor:
            mean_history.append(np.mean(history[:i]))
        else:
            mean_history.append(np.mean(history[i - smooth_factor : i]))
    plt.plot(mean_history)
plt.show()

#for seed in env.env_scores:
#    print('seed={}, score={}'.format(seed, np.mean(env.env_scores[seed])))