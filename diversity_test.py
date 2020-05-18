from metrics.dmetric import *

for base in range(10):
    env_list = []

    for i in [100 + base * 5, 101 + base * 5, 102 + base * 5, 103 + base * 5, 104 + base * 5]:
        env_list.append(Gridworld(width=4, height=4, cell_size=32, seed=i))

    for _ in range(1):
        diversity_list = get_diversity(env_list, training_iterations=5000)
        diversity = np.mean(diversity_list)
        print('base={}, diversity={}'.format(base, diversity))