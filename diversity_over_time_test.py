from metrics.dmetric_over_time import *
from util.experiment import Experiment

exp = Experiment('diversity_over_time_set_B_30')

env_list = []

for i in [115, 116, 117, 118, 119]:
    env_list.append(Gridworld(width=4, height=4, cell_size=32, seed=i))

exp.run(get_diversity, params={ 'env_list' : env_list, 'training_iterations' : 100, 'steps' : 30, 'verbose' : True }, k=3)