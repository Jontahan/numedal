from metrics.dmetric_reinforce import *
from util.experiment import Experiment

exp = Experiment('diversity_set_b_reinforce_5k', dry=False)

env_list = []

for i in range(10, 15):
    env_list.append(Gridworld(width=4, height=4, cell_size=32, seed=i))

exp.run(get_diversity, params={ 'env_list' : env_list, 'training_iterations' : 5000 }, k=10)