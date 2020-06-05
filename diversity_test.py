from metrics.diversity import get_diversity
from util.experiment import Experiment
from gw_collect import Gridworld

for base in range(1, 6):
    env_list = []

    for i in [base * 5, base  * 5 + 1, base  * 5 + 2, base  * 5 + 3, base  * 5 + 4]:
        env_list.append(Gridworld(width=4, height=4, cell_size=32, seed=i))

    # dqn full
    try:
        exp = Experiment('set{}_div_dqn_full'.format(base), dry=False)
        exp.run(get_diversity, params={ 'env_list' : env_list,
                                        'training_iterations' : 100,
                                        'steps' : 100,
                                        'verbose' : True,
                                        'learning_alg' : 'dqn',
                                        'state_dist' : 'full'
                                        }, k=3)
    except Exception as e: 
        print(e)

    # dqn memory
    try:
        exp = Experiment('set{}_div_dqn_memory'.format(base), dry=False)
        exp.run(get_diversity, params={ 'env_list' : env_list,
                                        'training_iterations' : 100,
                                        'steps' : 100,
                                        'verbose' : True,
                                        'learning_alg' : 'dqn',
                                        'state_dist' : 'memory'
                                        }, k=3)
    except Exception as e: 
        print(e)
    
    # soft dqn
    try:
        exp = Experiment('set{}_div_softdqn_full'.format(base), dry=False)
        exp.run(get_diversity, params={ 'env_list' : env_list,
                                        'training_iterations' : 100,
                                        'steps' : 100,
                                        'verbose' : True,
                                        'learning_alg' : 'dqn',
                                        'state_dist' : 'full',
                                        'softmax' : True
                                        }, k=3)
    except Exception as e: 
        print(e)

    # rein full
    try:
        exp = Experiment('set{}_div_rein_full'.format(base), dry=False)
        exp.run(get_diversity, params={ 'env_list' : env_list,
                                        'training_iterations' : 100,
                                        'steps' : 150,
                                        'verbose' : True,
                                        'learning_alg' : 'reinforce',
                                        'state_dist' : 'full'
                                        }, k=3)
    except Exception as e: 
        print(e)

    # rein memory
    try:
        exp = Experiment('set{}_div_rein_memory'.format(base), dry=False)
        exp.run(get_diversity, params={ 'env_list' : env_list,
                                        'training_iterations' : 100,
                                        'steps' : 150,
                                        'verbose' : True,
                                        'learning_alg' : 'reinforce',
                                        'state_dist' : 'memory'
                                        }, k=3)
    except Exception as e: 
        print(e)