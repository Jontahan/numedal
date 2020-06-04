from gw_collect import Gridworld
import pygame as pg

for i in range(10):
    env = Gridworld(width=4, height=4, cell_size=32, seed=i)
    env.reset()

    pg.init()
    screen = pg.display.set_mode((env.cell_size * env.width, env.cell_size * env.height))
    env.screen = screen
    env.draw(screen)