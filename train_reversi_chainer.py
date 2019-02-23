# -*- coding: utf-8 -*-
"""
リバーシプログラム（盤面8x8）：エージェント学習プログラム（CNN，DQNを利用）
Copyright(c) 2018 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
"""
from __future__ import print_function

from enviroment import Enviroment
from agent_chainer import Agent

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
import sys
import re # 正規表現
import random
import copy

# 定数定義 #
PLAYER_1 = 1  # ボードのある座標にある石：黒
PLAYER_2 = 2  # ボードのある座標にある石：白
REWARD_WIN = 1 # 勝ったときの報酬
REWARD_LOSE = -1 # 負けたときの報酬
    
# メイン関数            
def main():

    # 環境, 
    env = Enviroment()    

    # エージェント．agent_1用・agent_2用のエージェントを別々に学習する．DQNを利用．バッチサイズを少し大きめに設定
    agent_1 = Agent(env.obs_size, env.n_actions)
    agent_2 = Agent(env.obs_size, env.n_actions)

    agents = ['', agent_1, agent_2]
    
    n_episodes = 20000 # 最大のゲーム回数
    win = 0     # agent_1 の勝利数    
    lose = 0    # agent_1 の敗北数
    draw = 0    # 引き分け    

    # ゲーム開始（エピソードの繰り返し実行）
    for i in range(1, n_episodes + 1):
        env.reset()
        #print('DEBUG epi {} {}'.format(i+1, len(replay_buffer_b.memory)))
        rewards = [0, 0, 0] # 報酬リセット

        while not env.game_end: # ゲームが終わるまで繰り返す
            #print('DEBUG: rewards {}'.format(rewards))
            # 石が置けない場合はパス
            if not env.available_pos:
                env.pss += 1
                env.end_check()
            else:
                # 石を配置する場所を取得．
                while True: # 置ける場所が見つかるまで繰り返す
                    action = agents[env.turn].update_Q_table(env.state(), env.available_index(), rewards[env.turn])
                    pos = env.pos(action) # 座標を2次元（i,j）に変換
                    if env.is_available(pos):                        
                        break
                    else:
                        rewards[env.turn] = REWARD_LOSE # 石が置けない場所であれば負の報酬                                        
                # 石を配置
                env.agent_action(pos)
                if env.pss == 1: # 石が配置できた場合にはパスフラグをリセットしておく（双方が連続パスするとゲーム終了する）
                    env.pss = 0 

            # ゲーム時の処理
            if env.game_end:                                
                if env.winner == PLAYER_1:
                    rewards[PLAYER_1] = REWARD_WIN  # agent_1の勝ち報酬
                    rewards[PLAYER_2] = REWARD_LOSE # agent_2の負け報酬
                    win += 1
                elif env.winner == 0:
                    draw += 1                    
                else:
                    rewards[PLAYER_1] = REWARD_LOSE
                    rewards[PLAYER_2] = REWARD_WIN
                    lose += 1                    
                #エピソードを終了して学習                                
                # 勝者のエージェントの学習
                agents[env.turn].update_Q_table(env.state(), env.available_index(), rewards[env.turn], True)
                env.change_turn()
                # 敗者のエージェントの学習
                agents[env.turn].update_Q_table(env.state(), env.available_index(), rewards[env.turn], True)
            else:                
                env.change_turn()

        # 学習の進捗表示
        if i % 100 == 0:            
            print('==== Episode {} : black win {}, black lose {}, draw {} ===='.format(i, win, lose, draw)) # 勝敗数はagent_1基準
            print('<PLAYER_1> statistics: {}, epsilon {}'.format(agent_1.get_statistics(), agent_1.explorer.epsilon))
            print('<PLAYER_2> statistics: {}, epsilon {}'.format(agent_2.get_statistics(), agent_2.explorer.epsilon))
            # カウンタ変数の初期化            
            win = 0
            lose = 0
            draw = 0            

        if i % 1000 == 0: # 1000エピソードごとにモデルを保存する
            agent_1.save("agent_1_" + str(i))
            agent_2.save("agent_2_" + str(i))

if __name__ == '__main__':
    main()
