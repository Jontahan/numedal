from metrics.dmetric import *
from util.experiment import Experiment

for base in range(1, 100):
    exp = Experiment('diversity_set_size_10_offset_{}_3k'.format(base))

    env_list = []

    for i in range(10):
        env_list.append(Gridworld(width=4, height=4, cell_size=32, seed=(10 * base + i)))

    exp.run(get_diversity, params={ 'env_list' : env_list, 'training_iterations' : 3000 }, k=4)