from metrics.value_diff_function import *
from gw_collect import Gridworld

envs = [
    (Gridworld(width=4, height=4, cell_size=32, agent_pos=(2, 0), food_pos=[(0, 3), (3, 3)]),
     Gridworld(width=4, height=4, cell_size=32, agent_pos=(2, 0), food_pos=[(1, 3), (3, 3)]))
]

for env_pair in envs:
    print(env_diff(env_pair[0], env_pair[1], 10, 10))

for diff in absolutely_all_diffs:
    plt.plot(diff)

plt.savefig('test.png')
plt.show()