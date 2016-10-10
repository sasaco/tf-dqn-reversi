from __future__ import division

import argparse
import os

import matplotlib.pyplot as plt

from Reversi import Reversi
from dqn_agent import DQNAgent
  
env = Reversi()
env.print_screen()
"""
print("----------- エージェント -----------------")
agent = DQNAgent(env.enable_actions, env.name, env.screen_n_rows, env.screen_n_cols)
print(agent.enable_actions)
state, reward, terminal = env.observe()
#print(agent.Q_values[state])

action = agent.select_action(state, agent.exploration)
"""

print("----------- 1手目 -----------------")
pos = 43
print("○ pos={:}".format(pos))
env.put_piece(pos, 1)
env.print_screen()

print("----------- 2手目 -----------------")
pos = 44
print("● pos={:}".format(pos))
env.put_piece(pos, 2)
env.print_screen()


print("----------- 3手目 -----------------")
pos = 21
print("○ pos={:}".format(pos))
env.put_piece(pos, 1)
env.print_screen()

print("----------- 4手目 -----------------")
pos = 26
print("● pos={:}".format(pos))
env.put_piece(pos, 2)
env.print_screen()


print(env.get_enables(1))
print("----------- 5手目 -----------------")
pos = 34
print("○ pos={:}".format(pos))
env.put_piece(pos, 1)
env.print_screen()

print("----------- 6手目 -----------------")
pos = 61
print("● pos={:}".format(pos))
env.put_piece(pos, 2)
env.print_screen()

print("----------- 7手目 -----------------")
pos = 22
print("○ pos={:}".format(pos))
env.put_piece(pos, 1)
env.print_screen()



print("----------- 8手目 -----------------")
pos = 13
print("● pos={:}".format(pos))
env.put_piece(pos, 2)
env.print_screen()









