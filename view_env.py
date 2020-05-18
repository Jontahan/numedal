from gw_collect import Gridworld

env = Gridworld(width=4, height=4, cell_size=32, seed=103)
print(env.reset())