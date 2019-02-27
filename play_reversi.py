# -*- coding: utf-8 -*-
"""
リバーシプログラム（盤面8x8）：人間と対戦用プログラム（DQNを利用）
"""
from __future__ import print_function

from enviroment import Enviroment
#from agent_chainer import Agent
from agent_tensorflow import Agent

import numpy as np
import re # 正規表現

# 定数定義 #
ROWLABEL = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8} # ボードの横軸ラベル
N2L = ['', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] # 横軸ラベルの逆引き用


# キーボードから入力した座標を2次元配列に対応するよう変換する
def convert_coordinate(pos):
    pos = pos.split(' ')
    i = int(pos[0]) - 1
    j = int(ROWLABEL[pos[1]]) - 1
    return (i, j) # iが縦，jが横，タプルで返す
    
### メイン関数  ###
def main():

    # 環境, 
    env = Enviroment()    # ボードの初期化    

    # エージェント．DQNを利用．
    agent = Agent(env.obs_size, env.n_actions)
    
    ### ここからゲームスタート ###
    print('=== リバーシ ===')
    you = input('先攻（黒石, {}） or 後攻（白石, {}）を選択：'.format(env.BLACK, env.WHITE))
    you = int(you)
    trn = you    
    assert(you == env.BLACK or you == env.WHITE)
    level = input('難易度（弱 1〜10 強）：')
    level = int(level) * 2000

    if you == env.BLACK:
        s = '「●」（先攻）' 
        file = 'agent_2_' + str(level)        
        a = env.WHITE
    else:
        s = '「○」（後攻）'
        file = 'agent_1_' + str(level)      
        a = env.BLACK

    # 学習済のモデルをロードする
    agent.load(file) 

    print('あなたは{}です。ゲームスタート！'.format(s))
    env.print_state()

    
    # ゲーム開始
    while not env.game_end:

        if env.turn == a:            
            # エージェントのターン
            print('エージェントのターン --> ', end='')
            if not env.available_pos:
                env.pss += 1
                print('パスします。{}'.format(env.pss))
            else:
                action = agent.act(env.state(), env.available_index())
                pos = env.pos(action) # 座標を2次元（i,j）に変換
                if not env.is_available(pos):   
                    # 反則負け
                    print('エージェントが({},{})に置きました 反則です あなたの勝ち'.format(pos[0]+1, N2L[pos[1]+1]))
                    break
                env.agent_action(pos) # posに石を置く
                env.pss = 0
                print('({},{})'.format(pos[0]+1, N2L[pos[1]+1]))

            env.print_state()

        else:            
            # あなたのターン。
            while True:
                print('あなたのターン。')
                if not env.search_positions():
                    print('パスします。')
                    env.pss += 1
                else:
                    pos = input('どこに石を置きますか？ (行列で指定。例 "4 d")：')
                    if not re.match(r'[0-9] [a-z]', pos):
                        print('正しく座標を入力してください。')
                        continue
                    else:
                        if not env.is_available(convert_coordinate(pos)): # 置けない場所に置いた場合
                            print('ここには石を置けません。')
                            continue
                        env.agent_action(convert_coordinate(pos))
                        env.print_state()
                        env.pss = 0
                break

        # ゲーム終了チェック
        if env.game_end:
            if env.winner == a:
                print('Game over. You lose!')
            elif env.winner == you:
                print('Game over. You win！')
            else:
                print('Game over. Draw.')
            continue

        env.change_turn()  

if __name__ == '__main__':
    main()
