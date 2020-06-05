from metrics.dmetric_over_time import *
from util.experiment import Experiment

exp = Experiment('set_size__diversity', dry=True)

env_list = []
    
for j in [0, 1, 2, 3, 4]:
    env_list.append(Gridworld(width=4, height=4, cell_size=32, seed=j))

exp.run(get_diversity, params={ 'env_list' : env_list, 'training_iterations' : 100, 'steps' : 50, 'verbose' : True }, k=5)