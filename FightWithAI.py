
import argparse
import os
import numpy as np

from Reversi import Reversi
from dqn_agent import DQNAgent


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path")
    parser.add_argument("-s", "--save", dest="save", action="store_true")
    parser.set_defaults(save=False)
    args = parser.parse_args()

    # environmet, agent
    env = Reversi()
    agent = DQNAgent(env.enable_actions, env.name, env.screen_n_rows, env.screen_n_cols)
    agent.load_model(args.model_path)

    # game
    print("------------- GAME START ---------------")
    while not env.isEnd():
        print("*** userターン○ ***")
        env.print_screen()
        enables = env.get_enables(1)
        if len(enables) > 0:
            flg = False
            while not flg:
                print("番号を入力してください")
                print(enables)
                inp = input('>>>  ')
                action_t = int(inp)
                for i in enables:                
                    if action_t == i:
                        flg = True                       
                        break
                
            env.execute_action(action_t, 1)
        else:
            print("パス")
            
            
        if env.isEnd() == True:break
            
        print("*** AIターン● ***")
        env.print_screen()
        enables = env.get_enables(2)
        if len(enables) > 0:
            flg = False
            while not flg:
                action_t = agent.select_action(env.screen, None, 0.0)
                print('>>>  {:}'.format(action_t))              
                for i in enables:                
                    if action_t == i:
                        flg = True                       
                        break
            env.execute_action(action_t, 2)
        else:
            print("パス")


    print("*** ゲーム終了 ***")
    if env.winner() == 1:
        print("あなたの勝ち！ スコアは、{:}です。".format(env.get_score(1)))
    else:
        print("あなたの負け！ AIのスコアは、{:}です。".format(env.get_score(2)))
    
