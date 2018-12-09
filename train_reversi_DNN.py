# -*- coding: utf-8 -*-
"""
リバーシプログラム：エージェント学習プログラム（DNN，DQNを利用）
Copyright(c) 2018 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
"""
from __future__ import print_function
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
SIZE = 4   # ボードサイズ SIZE*SIZE
NONE = 0   # ボードのある座標にある石：なし
BLACK = 1  # ボードのある座標にある石：黒
WHITE = 2  # ボードのある座標にある石：白
STONE = [' ', '●', '○'] # 石の表示用
ROWLABEL = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8} # ボードの横軸ラベル
N2L = ['', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] # 横軸ラベルの逆引き用
REWARD_WIN = 1 # 勝ったときの報酬
REWARD_LOSE = -1 # 負けたときの報酬
# 2次元のボード上での隣接8方向の定義（左から，上，右上，右，右下，下，左下，左，左上）
DIR = ((-1,0), (-1,1), (0,1), (1,1), (1,0), (1, -1), (0,-1), (-1,-1))

### Q関数の定義 ###
class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_nodes):
        w = chainer.initializers.HeNormal(scale=1.0) # 重みの初期化
        super(QFunction, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(obs_size, n_nodes, initialW=w)
            self.l2 = L.Linear(n_nodes, n_nodes, initialW=w)
            self.l3 = L.Linear(n_nodes, n_nodes, initialW=w)
            self.l4 = L.Linear(n_nodes, n_actions, initialW=w)
    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))        
        return chainerrl.action_value.DiscreteActionValue(self.l4(h))

### リバーシボードクラス ###
class Board():    

    # インスタンス（最初はボードの初期化）
    def __init__(self):
        self.board_reset()        

    # ボードの初期化
    def board_reset(self):
        self.board = np.zeros((SIZE, SIZE), dtype=np.float32) # 全ての石をクリア．ボードは2次元配列（i, j）で定義する．
        mid = SIZE // 2 # 真ん中の基準ポジション
        # 初期4つの石を配置
        self.board[mid, mid] = WHITE
        self.board[mid-1, mid-1] = WHITE
        self.board[mid-1, mid] = BLACK
        self.board[mid, mid-1] = BLACK
        self.winner = NONE # 勝者
        self.turn = BLACK  # 黒石スタート
        self.game_end = False # ゲーム終了チェックフラグ
        self.pss = 0 # パスチェック用フラグ．双方がパスをするとゲーム終了
        self.nofb = 0 # ボード上の黒石の数
        self.nofw = 0 # ボード上の白石の数
        self.available_pos = self.search_positions() # self.turnの石が置ける場所のリスト

    # 石を置く&リバース処理
    def put_stone(self, pos):
        if self.is_available(pos):
            self.board[pos[0], pos[1]] = self.turn
            self.do_reverse(pos) # リバース
            return True
        else:           
            return False

    # ターンチェンジ
    def change_turn(self):
        self.turn = WHITE if self.turn == BLACK else BLACK
        self.available_pos = self.search_positions() # 石が置ける場所を探索しておく

    # ランダムに石を置く場所を決める（ε-greedy用）
    def random_action(self):
        if len(self.available_pos) > 0:
            pos = random.choice(self.available_pos) # 置く場所をランダムに決める
            pos = pos[0] * SIZE + pos[1] # 1次元座標に変換（NNの教師データは1次元でないといけない）
            return pos
        return False # 置く場所なし

    # エージェントのアクションと勝敗判定．置けない場所に置いたら負けとする．
    def agent_action(self, pos):
        self.put_stone(pos)
        self.end_check() # 石が置けたら，ゲーム終了をチェック

    # リバース処理
    def do_reverse(self, pos):
        for di, dj in DIR:
            opp = BLACK if self.turn == WHITE else WHITE # 対戦相手の石
            boardcopy = self.board.copy() # 一旦ボードをコピーする（copyを使わないと参照渡しになるので注意）
            i = pos[0]
            j = pos[1]
            flag = False # 挟み判定用フラグ
            while 0 <= i < SIZE and 0 <= j < SIZE: # (i,j)座標が盤面内に収まっている間繰り返す
                i += di # i座標（縦）をずらす
                j += dj # j座標（横）をずらす
                if 0 <= i < SIZE and 0 <= j < SIZE and boardcopy[i,j] == opp:  # 盤面に収まっており，かつ相手の石だったら
                    flag = True
                    boardcopy[i,j] = self.turn # 自分の石にひっくり返す
                elif not(0 <= i < SIZE and 0 <= j < SIZE) or (flag == False and boardcopy[i,j] != opp):
                    break
                elif boardcopy[i,j] == self.turn and flag == True: # 自分と同じ色の石がくれば挟んでいるのでリバース処理を確定
                    self.board = boardcopy.copy() # ボードを更新
                    break

    # 石が置ける場所をリストアップする．石が置ける場所がなければ「パス」となる
    def search_positions(self):
        pos = []
        emp = np.where(self.board == 0) # 石が置かれていない場所を取得
        for i in range(emp[0].size): # 石が置かれていない全ての座標に対して
            p = (emp[0][i], emp[1][i]) # (i,j)座標に変換
            if self.is_available(p):
                pos.append(p) # 石が置ける場所の座標リストの生成
        return pos

    # 石が置けるかをチェックする
    def is_available(self, pos):
        if self.board[pos[0], pos[1]] != NONE: # 既に石が置いてあれば，置けない
            return False
        opp = BLACK if self.turn == WHITE else WHITE
        for di, dj in DIR: # 8方向の挟み（リバースできるか）チェック
            i = pos[0]
            j = pos[1]
            flag = False # 挟み判定用フラグ
            while 0 <= i < SIZE and 0 <= j < SIZE: # (i,j)座標が盤面内に収まっている間繰り返す
                i += di # i座標（縦）をずらす
                j += dj # j座標（横）をずらす
                if 0 <= i < SIZE and 0 <= j < SIZE and self.board[i,j] == opp: #盤面に収まっており，かつ相手の石だったら
                    flag = True
                elif not(0 <= i < SIZE and 0 <= j < SIZE) or (flag == False and self.board[i,j] != opp) or self.board[i,j] == NONE:                
                    break
                elif self.board[i,j] == self.turn and flag == True: # 自分と同じ色の石                    
                    return True
        return False
        
    # ゲーム終了チェック
    def end_check(self):
        if np.count_nonzero(self.board) == SIZE * SIZE or self.pss == 2: # ボードに全て石が埋まるか，双方がパスがしたら
            self.game_end = True
            self.nofb = len(np.where(self.board==BLACK)[0])
            self.nofw = len(np.where(self.board==WHITE)[0])
            self.winner = BLACK if len(np.where(self.board==BLACK)[0]) > len(np.where(self.board==WHITE)[0]) else WHITE
    
# メイン関数            
def main():
    board = Board() # ボード初期化    
    
    obs_size = SIZE * SIZE # ボードサイズ（=NN入力次元数）
    n_actions = SIZE * SIZE # 行動数はSIZE*SIZE（ボードのどこに石を置くか）
    n_nodes = 256 # 中間層のノード数
    q_func = QFunction(obs_size, n_actions, n_nodes)
    
    # optimizerの設定
    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(q_func)
    # 減衰率
    gamma = 0.99
    # ε-greedy法
    explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
        start_epsilon=1.0, end_epsilon=0.1, decay_steps=50000, random_action_func=board.random_action)
    # Expericence Replay用のバッファ（十分大きく，エージェント毎に用意）
    replay_buffer_b = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
    replay_buffer_w = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
    # エージェント．黒石用・白石用のエージェントを別々に学習する．DQNを利用．バッチサイズを少し大きめに設定
    agent_black = chainerrl.agents.DQN(
        q_func, optimizer, replay_buffer_b, gamma, explorer, 
        replay_start_size=1000, minibatch_size=128, update_interval=1, target_update_interval=1000)
    agent_white = chainerrl.agents.DQN(
        q_func, optimizer, replay_buffer_w, gamma, explorer,
        replay_start_size=1000, minibatch_size=128, update_interval=1, target_update_interval=1000)
    agents = ['', agent_black, agent_white]
    
    n_episodes = 20000 # 学習ゲーム回数
    win = 0 # 黒の勝利数    
    lose = 0 # 黒の敗北数
    draw = 0 # 引き分け    

    # ゲーム開始（エピソードの繰り返し実行）
    for i in range(1, n_episodes + 1):
        board.board_reset()
        rewards = [0, 0, 0] # 報酬リセット

        while not board.game_end: # ゲームが終わるまで繰り返す
            #print('DEBUG: rewards {}'.format(rewards))
            # 石が置けない場合はパス
            if not board.available_pos:
                board.pss += 1
                board.end_check()
            else:
                # 石を配置する場所を取得．ボードは2次元だが，NNへの入力のため1次元に変換．
                boardcopy = np.reshape(board.board.copy(), (-1,))
                while True: # 置ける場所が見つかるまで繰り返す
                    pos = agents[board.turn].act_and_train(boardcopy, rewards[board.turn])
                    pos = divmod(pos, SIZE) # 座標を2次元（i,j）に変換
                    if board.is_available(pos):                        
                        break
                    else:
                        rewards[board.turn] = REWARD_LOSE # 石が置けない場所であれば負の報酬                                        
                # 石を配置
                board.agent_action(pos)
                if board.pss == 1: # 石が配置できた場合にはパスフラグをリセットしておく（双方が連続パスするとゲーム終了する）
                    board.pss = 0 

            # ゲーム時の処理
            if board.game_end:                                
                if board.winner == BLACK:
                    rewards[BLACK] = REWARD_WIN  # 黒の勝ち報酬
                    rewards[WHITE] = REWARD_LOSE # 白の負け報酬
                    win += 1
                elif board.winner == 0:
                    draw += 1                    
                else:
                    rewards[BLACK] = REWARD_LOSE
                    rewards[WHITE] = REWARD_WIN
                    lose += 1                    
                #エピソードを終了して学習                
                boardcopy = np.reshape(board.board.copy(), (-1,))
                # 勝者のエージェントの学習
                agents[board.turn].stop_episode_and_train(boardcopy, rewards[board.turn], True)
                board.change_turn()
                # 敗者のエージェントの学習
                agents[board.turn].stop_episode_and_train(boardcopy, rewards[board.turn], True) 
            else:                
                board.change_turn()

        # 学習の進捗表示 (100エピソードごと)
        if i % 100 == 0:            
            print('==== Episode {} : black win {}, black lose {}, draw {} ===='.format(i, win, lose, draw)) # 勝敗数は黒石基準
            print('<BLACK> statistics: {}, epsilon {}'.format(agent_black.get_statistics(), agent_black.explorer.epsilon))
            print('<WHITE> statistics: {}, epsilon {}'.format(agent_white.get_statistics(), agent_white.explorer.epsilon))
            # カウンタ変数の初期化            
            win = 0
            lose = 0
            draw = 0            

        if i % 1000 == 0: # 1000エピソードごとにモデルを保存する
            agent_black.save("agent_black_" + str(i))
            agent_white.save("agent_white_" + str(i))

if __name__ == '__main__':
    main()
