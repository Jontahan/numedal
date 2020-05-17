from metrics.dmetric import *

env_list = []

for i in [8, 0, 2, 4, 7]:
    env_list.append(Gridworld(width=4, height=4, cell_size=32, seed=i))

for _ in range(10):
    diversity_list = get_diversity(env_list)
    diversity = np.mean(diversity_list)
    print(diversity)