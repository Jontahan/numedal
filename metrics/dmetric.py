from .value_diff_function import *
import time

def get_diversity(env_list, training_iterations=1000):
    dqn_list = []
    
    for i in range(len(env_list)):
        dqn = DQN(env_list[i], qnet=LinRegNet(64, 4).double(), plotter=None, render=False, memory_length=2000, gamma=.99, alpha=.001, epsilon_start=0.3)
        print('Training value function for environment({}) {}/{}'.format(env_list[i].seed, i + 1, len(env_list)))
        start = time.time()
        dqn.train(training_iterations, 4)
        end = time.time()
        print('Elapsed time: {:.2f}s'.format(end - start))
        dqn_list.append(dqn)
    
    diff_list = []
    for i in range(len(env_list)):
        for j in range(i + 1, len(env_list)):
            env_a = env_list[i]
            env_b = env_list[j]
            dqn_a = dqn_list[i]
            dqn_b = dqn_list[j]
            
            all_states = env_a.get_all_states() + env_b.get_all_states()
            
            env_diffs = []

            for state in all_states:
                state = torch.from_numpy(state)
                env_diffs.append(torch.sum((dqn_a.qnet(state) - dqn_b.qnet(state)) ** 2).item())

            diff_list.append(np.mean(env_diffs))
    
    
    return diff_list