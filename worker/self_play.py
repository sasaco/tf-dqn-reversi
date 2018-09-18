import copy

from datas.MoveHistory import MoveHistory

class self_play:

    def __init__(self, env, player: dict):
        self.env = env
        self.player = player


    def start(self, learning_target_player: int, game_idx: int,  save = True):

        n_epochs = 50

        move_list = []
        win_count = 0
        i = 0

        self.player[learning_target_player].exploration = 0.9

        while win_count < n_epochs:
            i += 1

            # 1ゲーム開始
            move = self.start_game(learning_target_player)

            # for log 勝率を計算する
            expl = self.player[learning_target_player].exploration
            if self.env.winner() == learning_target_player:
                win_count += 1
                self.player[learning_target_player].exploration = min(expl+0.01,0.9)
            else:
                self.player[learning_target_player].exploration = max(expl-0.05,0.4)


            # 行動を学習対象として保存
            move_list.extend(move)

            print("{} | TARGET: {:01d} | self_play | EXP: {:.4f} | EPOCH: {} | WIN: {}/{}".format(
                               game_idx, 
                               learning_target_player, 
                               self.player[learning_target_player].exploration,
                               i, win_count, n_epochs ))

        # 行動を学習対象として保存
        if save == True:
            MoveHistory.save_play_data(learning_target_player, move_list)

        return move_list


    def start_game(self, learning_target_player):

        # 行動を記録する変数を用意
        move = []
        
        self.env.reset()
        buffer = None

        while not self.env.isEnd(): # 1エピソードが終わるまでループ

            state = self.env.screen

            # 全候補手
            targets = self.env.get_enables(self.env.next_player) 

            # 行動 を バッファに記録
            if buffer != None:
                if learning_target_player == self.env.next_player:
                    # 前回の状態に今回の状態を足して保存
                    buffer.state_1 = copy.copy(self.env.screen)
                    buffer.targets_1 = copy.copy(targets)
                    buffer.reward = 0
                    buffer.terminal = False
                    move.append(buffer)
                    buffer = None

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

            # 行動を実行
            self.env.update(action, self.env.next_player) 

            # ゲームが終わったら最終状態 を バッファに記録
            if self.env.isEnd() == True:
                buffer.terminal = True
                if self.env.winner() == learning_target_player:
                    buffer.reward = 1 
                else:
                    buffer.reward = -1 
                move.append(buffer)

        return move


