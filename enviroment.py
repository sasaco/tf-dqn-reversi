import numpy as np

### リバーシボードクラス ###
class Enviroment:    

    # インスタンス（最初はボードの初期化）
    def __init__(self):

        # 定数定義 #
        self.SIZE = 8   # ボードサイズ SIZE*SIZE
        self.NONE = 0   # ボードのある座標にある石：なし
        self.BLACK = 1  # ボードのある座標にある石：黒
        self.WHITE = 2  # ボードのある座標にある石：白
        self.REWARD_WIN = 1 # 勝ったときの報酬
        self.REWARD_LOSE = -1 # 負けたときの報酬
        # 2次元のボード上での隣接8方向の定義（左から，上，右上，右，右下，下，左下，左，左上）
        self.DIR = ((-1,0), (-1,1), (0,1), (1,1), (1,0), (1, -1), (0,-1), (-1,-1))
        # ボードの初期化
        self.reset()        

        self.obs_size = self.SIZE * self.SIZE      # ボードサイズ（=NN入力次元数）
        self.n_actions = self.SIZE * self.SIZE     # 行動数はSIZE*SIZE（ボードのどこに石を置くか）

    # ボードの初期化
    def reset(self):
        self.board = np.zeros((self.SIZE, self.SIZE), dtype=np.float32) # 全ての石をクリア．ボードは2次元配列（i, j）で定義する．
        mid = self.SIZE // 2 # 真ん中の基準ポジション
        # 初期4つの石を配置
        self.board[mid, mid] = self.WHITE
        self.board[mid-1, mid-1] = self.WHITE
        self.board[mid-1, mid] = self.BLACK
        self.board[mid, mid-1] = self.BLACK
        self.winner = self.NONE # 勝者
        self.turn = self.BLACK  # 黒石スタート
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

    def available_index(self):

        index = []
        # 石が置かれていない場所を取得
        pos = self.search_positions()
        for p in pos:
            i = p[0] * self.SIZE + p[1]
            index.append(i) # 石が置ける場所の座標リストの生成

        return index


    # ターンチェンジ
    def change_turn(self):
        self.turn = self.WHITE if self.turn == self.BLACK else self.BLACK
        self.available_pos = self.search_positions() # 石が置ける場所を探索しておく

    def state(self):
         # 石を配置する場所を取得．ボードは2次元だが，NNへの入力のため1次元に変換．
        return  np.reshape(self.board.copy(), (-1,))

    def pos(self, index):
        # 座標を2次元（i,j）に変換
        return divmod(index, self.SIZE)

    # エージェントのアクションと勝敗判定．
    def agent_action(self, pos):
        self.put_stone(pos)
        self.end_check()


    # リバース処理
    def do_reverse(self, pos):
        for di, dj in self.DIR:
            opp = self.BLACK if self.turn == self.WHITE else self.WHITE # 対戦相手の石
            boardcopy = self.board.copy() # 一旦ボードをコピーする（copyを使わないと参照渡しになるので注意）
            i = pos[0]
            j = pos[1]
            flag = False # 挟み判定用フラグ
            while 0 <= i < self.SIZE and 0 <= j < self.SIZE: # (i,j)座標が盤面内に収まっている間繰り返す
                i += di # i座標（縦）をずらす
                j += dj # j座標（横）をずらす
                if 0 <= i < self.SIZE and 0 <= j < self.SIZE and boardcopy[i,j] == opp:  # 盤面に収まっており，かつ相手の石だったら
                    flag = True
                    boardcopy[i,j] = self.turn # 自分の石にひっくり返す
                elif not(0 <= i < self.SIZE and 0 <= j < self.SIZE) or (flag == False and boardcopy[i,j] != opp):
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
        if self.board[pos[0], pos[1]] != self.NONE: # 既に石が置いてあれば，置けない
            return False
        opp = self.BLACK if self.turn == self.WHITE else self.WHITE
        for di, dj in self.DIR: # 8方向の挟み（リバースできるか）チェック
            i = pos[0]
            j = pos[1]
            flag = False # 挟み判定用フラグ
            while 0 <= i < self.SIZE and 0 <= j < self.SIZE: # (i,j)座標が盤面内に収まっている間繰り返す
                i += di # i座標（縦）をずらす
                j += dj # j座標（横）をずらす
                if 0 <= i < self.SIZE and 0 <= j < self.SIZE and self.board[i,j] == opp: #盤面に収まっており，かつ相手の石だったら
                    flag = True
                elif not(0 <= i < self.SIZE and 0 <= j < self.SIZE) or (flag == False and self.board[i,j] != opp) or self.board[i,j] == self.NONE:                
                    break
                elif self.board[i,j] == self.turn and flag == True: # 自分と同じ色の石                    
                    return True
        return False
  
    
    # ゲーム終了チェック
    def end_check(self):
        if np.count_nonzero(self.board) == self.SIZE * self.SIZE or self.pss == 2: # ボードに全て石が埋まるか，双方がパスがしたら
            self.game_end = True
            self.nofb = len(np.where(self.board==self.BLACK)[0])
            self.nofw = len(np.where(self.board==self.WHITE)[0])
            self.winner = self.BLACK if len(np.where(self.board==self.BLACK)[0]) > len(np.where(self.board==self.WHITE)[0]) else self.WHITE
    