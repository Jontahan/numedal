from metrics.dmetric_over_time_rbuf import *
from util.experiment import Experiment

for i in range(0, 10):
    exp = Experiment('identity_test_{}_all_states_25k'.format(i))

    env_list = [ Gridworld(width=4, height=4, cell_size=32, seed=i),
                 Gridworld(width=4, height=4, cell_size=32, seed=i) ]

    exp.run(get_diversity, params={ 'env_list' : env_list, 'training_iterations' : 1000, 'steps' : 25, 'verbose' : True }, k=1)