import copy

from datas.MoveHistory import MoveHistory
from env.Reversi import Reversi

class self_play:

    def __init__(self, env: Reversi, player: dict):
        # Reversi 環境
        self.env = env
        # プレイヤーAI　２体
        self.player = player


    def start(self, learning_target_player: int, game_idx: int,  save = True):

        # 50 回学習する
        n_epochs = 50
        # 行動を保存する配列
        move_list = []
        #勝った回数
        win_count = 0
        #試合数
        i = 0

        # n_epochs 回勝つまでゲームする
        while win_count < n_epochs:
            i += 1

            # 1ゲーム開始
            move = self.start_game(learning_target_player)

            # for log 勝った回数を記録する
            if self.env.winner() == learning_target_player:
                win_count += 1

            # 行動を学習対象として保存
            move_list.extend(move)

            # 学習回数 | 学習対象プレーヤー|学習率|試合回|勝率
            print("{} | TARGET: {:01d} | self_play | EXP: {:.4f} | EPOCH: {} | WIN: {}/{}".format(
                               game_idx, 
                               learning_target_player, 
                               self.player[learning_target_player].exploration,
                               i, win_count, n_epochs ))

        # 行動を学習対象として保存
        if save == True:
            MoveHistory.save_play_data(learning_target_player, move_list)

        return move_list


    # 試合を1回行う
    def start_game(self, learning_target_player: int):

        # 行動を記録する変数を用意
        move = []
        # ゲーム盤面をリセット
        self.env.reset()
        # 記録用バッファ
        buffer = None

        # 1エピソードが終わるまでループ
        while not self.env.isEnd(): 

            state = self.env.screen

            # 全候補手
            targets = self.env.get_enables(self.env.next_player) 

            # ランダムで打つ確率
            if learning_target_player == self.env.next_player:
                # 学習対象なら一定の確率でランダムに手を選択する
                player_exploration = self.player[self.env.next_player].exploration 
            else:
                # 学習対象ではない(対戦相手として振る舞う)場合 学習結果から次の手を決定する
                player_exploration = -1

            # 行動を選択
            action = self.player[self.env.next_player].select_action(state, targets, player_exploration)
            
            # 行動 を バッファに記録
            if learning_target_player == self.env.next_player:
                buffer = MoveHistory()
                buffer.state = copy.copy(self.env.screen)
                buffer.targets = copy.copy(targets)
                buffer.action = action
            elif buffer != None:
                # 前回の状態に今回の状態を足して保存
                buffer.state_1 = copy.copy(self.env.screen)
                buffer.targets_1 = copy.copy(targets)
                # 相手がパスした場合
                if learning_target_player == self.env.next_player:
                    # 前回の状態に今回の状態を足して保存
                    buffer.reward = 0
                    buffer.terminal = False
                    move.append(buffer)
                    # 今回の状態を保存
                    buffer = MoveHistory()
                    buffer.state = copy.copy(self.env.screen)
                    buffer.targets = copy.copy(targets)
                    buffer.action = action
                   
            # 行動を実行
            self.env.update(action, self.env.next_player) 

            # ゲームが終わったら最終状態 を バッファに記録
            if buffer != None:
                if self.env.isEnd() == True:
                    buffer.terminal = True
                    if self.env.winner() == learning_target_player:
                        buffer.reward = 1
                    else:
                        buffer.reward = -1
                    move.append(buffer)
                else:
                    # ゲームの途中の手を保存
                    buffer.reward = 0
                    buffer.terminal = False
                    move.append(buffer)
                    buffer = None

        return move


