import numpy as np
from gw_collect import Gridworld
from util.experiment import Experiment

from common.nn import LinRegNet, LogRegNet
from vicero.algorithms.deepqlearning import DQN
from vicero.algorithms.reinforce import Reinforce

exp = Experiment('gw_collect_rein_benchmark_all2', dry=False)

def random_benchmark(env, iterations):
    all_durations = []
    
    for _ in range(iterations):
        env.reset()
        done = False
        duration = 0
        while not done:
            _, _, done, _ = env.step(env.action_space.sample())
            duration += 1
        all_durations.append(duration)
    
    return all_durations

def dqn_benchmark(env, iterations):
    histories = []
    for _ in range(iterations):
        dqn = DQN(env, qnet=LinRegNet(64, 4).double(), plot_durations=True, plotter=None, render=False, memory_length=2000, gamma=.99, alpha=.001, epsilon_start=0.1)
        dqn.train(200, 4)
        histories.append(np.mean(dqn.history))
    return histories

def rein_benchmark(env, iterations):
    histories = []
    for _ in range(iterations):
        try:
            rein = Reinforce(env, LogRegNet(64, 4).double())
            rein.train(200)
            histories.append(np.mean(rein.episode_history))
        except:
            print(':(')
    return histories

def average_environment_duration_random(env_set, iterations_per_env):
    all_durations = []
    for env in env_set:
        all_durations.append('env={},mean_duration={}'.format(env, np.mean(random_benchmark(env, iterations_per_env))))
        print(all_durations[-1])
    return all_durations

def average_environment_duration_dqn(env_set, iterations_per_env):
    all_durations = []
    for env in env_set:
        all_durations.append('env={},mean_duration={}'.format(env, np.mean(dqn_benchmark(env, iterations_per_env))))
        print(all_durations[-1])
    return all_durations


def average_environment_duration_rein(env_set, iterations_per_env):
    all_durations = []
    for env in env_set:
        all_durations.append('env={},mean_duration={}'.format(env, np.mean(rein_benchmark(env, iterations_per_env))))
        print(all_durations[-1])
    return all_durations

#env = Gridworld(width=4, height=4, cell_size=32, seed=33)
env_set = []
for i in range(10):
    env_set.append(Gridworld(width=4, height=4, cell_size=32, seed=i))

exp.run(average_environment_duration_rein, params={ 'env_set' : env_set, 'iterations_per_env' : 10 }, k=1)