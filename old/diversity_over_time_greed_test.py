from metrics.dmetric_over_time_greed import *
from util.experiment import Experiment

exp = Experiment('diversity_greed_test_setC')

env_list = []
    
for j in [10,11,12,13,14]:
    env_list.append(Gridworld(width=4, height=4, cell_size=32, seed=j))

exp.run(get_diversity, params={ 'env_list' : env_list, 'training_iterations' : 100, 'steps' : 50, 'verbose' : True }, k=5)