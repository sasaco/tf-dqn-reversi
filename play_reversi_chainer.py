# -*- coding: utf-8 -*-
"""
リバーシプログラム（盤面8x8）：人間と対戦用プログラム（CNN，DQNを利用）
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
SIZE = 8   # ボードサイズ SIZE*SIZE
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

#### Q関数の定義 ###
class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_nodes):
        w = chainer.initializers.HeNormal(scale=1.0) # 重みの初期化
        super(QFunction, self).__init__()
        with self.init_scope():
            self.c1 = L.Convolution2D(1, 4, 2, 1, 0)
            self.c2 = L.Convolution2D(4, 8, 2, 1, 0)
            self.c3 = L.Convolution2D(8, 16, 2, 1, 0)
            self.l4 = L.Linear(400, n_nodes, initialW=w)
            self.l5 = L.Linear(n_nodes, n_actions, initialW=w)

    # フォワード処理
    def __call__(self, x):
        #print('DEBUG: forward {}'.format(x))
        h = F.relu(self.c1(x))
        h = F.relu(self.c2(h))
        h = F.relu(self.c3(h))
        h = F.relu(self.l4(h))
        return chainerrl.action_value.DiscreteActionValue(self.l5(h))

#### リバーシボードクラス ###
class Board():    

    # インスタンス
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
            self.do_reverse(pos) # 石のリバース
            return True
        else:           
            return False

    # 白黒のターンチェンジ
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
    
    # エージェントのアクションと勝敗判定．
    def agent_action(self, pos):
        self.put_stone(pos)
        self.end_check() 

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
        emp = np.where(self.board == 0) # 石が置いていない場所を取得
        for i in range(emp[0].size):
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
            #import pdb; pdb.set_trace()
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

    # ボード表示
    def show_board(self):
        print('  ', end='')            
        for i in range(1, SIZE + 1):
            print(' {}'.format(N2L[i]), end='') # 横軸ラベル表示
        print('')
        for i in range(0, SIZE):
            print('{0:2d} '.format(i+1), end='')
            for j in range(0, SIZE):
                print('{} '.format(STONE[ int(self.board[i][j]) ]), end='') 
            print('')

# キーボードから入力した座標を2次元配列に対応するよう変換する
def convert_coordinate(pos):
    pos = pos.split(' ')
    i = int(pos[0]) - 1
    j = int(ROWLABEL[pos[1]]) - 1
    return (i, j) # iが縦，jが横，タプルで返す
    
### メイン関数  ###
def main():
    
    board = Board() # ボードの初期化    
    
    obs_size = SIZE * SIZE # ボードサイズ（DNN入力次元数）
    n_actions = SIZE * SIZE # 行動数はSIZE*SIZE（=ボードのどこに石を置くか）
    n_nodes = 256 # 中間層のノード数
    q_func = QFunction(obs_size, n_actions, n_nodes)
    
    # optimizerの設定
    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(q_func)
    # 減衰率
    gamma = 0.99
    # ε-greedy法
    explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
        start_epsilon=1.0, end_epsilon=0.1, decay_steps=100000, random_action_func=board.random_action)
    # Expericence Replay用のバッファ（十分大きく）
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
    # エージェント．DQNを利用．
    agent = chainerrl.agents.DQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        replay_start_size=1000, minibatch_size=128, update_interval=1, target_update_interval=1000)

    ### ここからゲームスタート ###
    print('=== リバーシ ===')
    you = input('先攻（黒石, 1） or 後攻（白石, 2）を選択：')
    you = int(you)
    trn = you    
    assert(you == BLACK or you == WHITE)
    level = input('難易度（弱 1〜10 強）：')
    level = int(level) * 2000
    if you == BLACK:
        s = '「●」（先攻）' 
        file = 'agent_white_' + str(level)        
        a = WHITE
    else:
        s = '「○」（後攻）'
        file = 'agent_black_' + str(level)      
        a = BLACK
    agent.load(file)
    print('あなたは{}です。ゲームスタート！'.format(s))
    board.show_board()
    
    # ゲーム開始
    while not board.game_end:
        if trn == 2:            
            boardcopy = np.reshape(board.board.copy(), (1,SIZE,SIZE))
            pos = divmod(agent.act(boardcopy), SIZE)
            if not board.is_available(pos): # NNで置く場所が置けない場所であれば置ける場所からランダムに選択する．
                pos = board.random_action()
                if not pos: # 置く場所がなければパス                    
                    board.pss += 1
                else:
                    pos = divmod(pos, SIZE) # 座標を2次元に変換
            print('エージェントのターン --> ', end='')
            if board.pss > 0 and not pos:
                print('パスします。{}'.format(board.pss))
            else:
                board.agent_action(pos) # posに石を置く
                board.pss = 0
                print('({},{})'.format(pos[0]+1, N2L[pos[1]+1]))
            board.show_board()
            board.end_check() # ゲーム終了チェック
            if board.game_end:
                if board.winner == a:
                    print('Game over. You lose!')
                elif board.winner == you:
                    print('Game over. You win！')
                else:
                    print('Game over. Draw.')
                continue
            board.change_turn() #　エージェント --> You

        while True:
            print('あなたのターン。')
            if not board.search_positions():
                print('パスします。')
                board.pss += 1
            else:
                pos = input('どこに石を置きますか？ (行列で指定。例 "4 d")：')
                if not re.match(r'[0-9] [a-z]', pos):
                    print('正しく座標を入力してください。')
                    continue
                else:
                    if not board.is_available(convert_coordinate(pos)): # 置けない場所に置いた場合
                        print('ここには石を置けません。')
                        continue
                    board.agent_action(convert_coordinate(pos))
                    board.show_board()
                    board.pss = 0
            break
        board.end_check()
        if board.game_end:
            if board.winner == you:
                print('Game over. You win!')
            elif board.winner == a:
                print('Game over. You lose.')
            else:
                print('Game over. Draw.')
            continue

        trn = 2
        board.change_turn()  

if __name__ == '__main__':
    main()
