from metrics.dmetric_reinforce_rbuf_over_time import *
from util.experiment import Experiment

for base in range(0, 10):
    try:
        exp = Experiment('diversity_over_time_rein_full_150steps_base{}'.format(base), dry=False)

        env_list = []

        for i in [base * 5, base  * 5 + 1, base  * 5 + 2, base  * 5 + 3, base  * 5 + 4]:
            env_list.append(Gridworld(width=4, height=4, cell_size=32, seed=i))

        exp.run(get_diversity, params={ 'env_list' : env_list, 'training_iterations' : 100, 'steps' : 150, 'verbose' : True }, k=2)
    except:
        print('crash base=', base)