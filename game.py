# ====================
# リバーシ
# ====================

# パッケージのインポート
import random
import math

# ゲーム状態
class State:
    # 初期化
    def __init__(self, pieces=None, enemy_pieces=None, depth=0):
        # 方向定数
        self.dxy = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))

        # 連続パスによる終了
        self.pass_end = False

        # 石の配置
        self.pieces = pieces
        self.enemy_pieces = enemy_pieces
        self.depth = depth

        # 石の初期配置
        if pieces == None or enemy_pieces == None:
            self.pieces = [0] * 36
            self.pieces[14] = self.pieces[21] = 1
            self.enemy_pieces = [0] * 36
            self.enemy_pieces[15] = self.enemy_pieces[20] = 1

    # 石の数の取得
    def piece_count(self, pieces):
        count = 0
        for i in pieces:
            if i == 1:
                count +=  1
        return count

    # 負けかどうか
    def is_lose(self):
        return self.is_done() and self.piece_count(self.pieces) < self.piece_count(self.enemy_pieces)

    # 引き分けかどうか
    def is_draw(self):
        return self.is_done() and self.piece_count(self.pieces) == self.piece_count(self.enemy_pieces)

    # ゲーム終了かどうか
    def is_done(self):
        return self.piece_count(self.pieces) + self.piece_count(self.enemy_pieces) == 36 or self.pass_end

    # 次の状態の取得
    def next(self, action):
        state = State(self.pieces.copy(), self.enemy_pieces.copy(), self.depth+1)
        if action != 36:
            state.is_legal_action_xy(action%6, int(action/6), True)
        w = state.pieces
        state.pieces = state.enemy_pieces
        state.enemy_pieces = w

        # 2回連続パス判定
        if action == 36 and state.legal_actions() == [36]:
            state.pass_end = True
        return state

    # 合法手のリストの取得
    def legal_actions(self):
        actions = []
        for j in range(0,6):
            for i in range(0,6):
                if self.is_legal_action_xy(i, j):
                    actions.append(i+j*6)
        if len(actions) == 0:
            actions.append(36) # パス
        return actions

    # 任意のマスが合法手かどうか
    def is_legal_action_xy(self, x, y, flip=False):
        # 任意のマスの任意の方向が合法手かどうか
        def is_legal_action_xy_dxy(x, y, dx, dy):
            # １つ目 相手の石
            x, y = x+dx, y+dy
            if y < 0 or 5 < y or x < 0 or 5 < x or \
                self.enemy_pieces[x+y*6] != 1:
                return False

            # 2つ目以降
            for j in range(6):
                # 空
                if y < 0 or 5 < y or x < 0 or 5 < x or \
                    (self.enemy_pieces[x+y*6] == 0 and self.pieces[x+y*6] == 0):
                    return False

                # 自分の石
                if self.pieces[x+y*6] == 1:
                    # 反転
                    if flip:
                        for i in range(6):
                            x, y = x-dx, y-dy
                            if self.pieces[x+y*6] == 1:
                                return True
                            self.pieces[x+y*6] = 1
                            self.enemy_pieces[x+y*6] = 0
                    return True
                # 相手の石
                x, y = x+dx, y+dy
            return False

        # 空きなし
        if self.enemy_pieces[x+y*6] == 1 or self.pieces[x+y*6] == 1:
            return False

        # 石を置く
        if flip:
            self.pieces[x+y*6] = 1

        # 任意の位置が合法手かどうか
        flag = False
        for dx, dy in self.dxy:
            if is_legal_action_xy_dxy(x, y, dx, dy):
                flag = True
        return flag

    # 先手かどうか
    def is_first_player(self):
        return self.depth%2 == 0

    # 文字列表示
    def __str__(self):
        ox = ('o', 'x') if self.is_first_player() else ('x', 'o')
        str = ''
        for i in range(36):
            if self.pieces[i] == 1:
                str += ox[0]
            elif self.enemy_pieces[i] == 1:
                str += ox[1]
            else:
                str += '-'
            if i % 6 == 5:
                str += '\n'
        return str

# ランダムで行動選択
def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions)-1)]

# 動作確認
if __name__ == '__main__':
    # 状態の生成
    state = State()

    # ゲーム終了までのループ
    while True:
        # ゲーム終了時
        if state.is_done():
            break

        # 次の状態の取得
        state = state.next(random_action(state))

        # 文字列表示
        print(state)
        print()