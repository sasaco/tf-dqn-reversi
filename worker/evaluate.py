

class evaluate:
    
    def __init__(self, env, player: dict):
        self.env = env
        self.player = player


    def start(self, learning_target_player: int):

        win_count = 0

        # 1ゲーム開始
        winner = self.start_game()

        if winner == learning_target_player:
            win_count += 1

        return win_count 


    def start_game(self):

        self.env.reset()

        while not self.env.isEnd(): # 1エピソードが終わるまでループ

            state = self.env.screen

            # 全候補手
            targets = self.env.get_enables(self.env.next_player) 

            # 学習結果から次の手を決定する
            player_exploration = -1

            # 行動を選択
            action = self.player[self.env.next_player].select_action(state, targets, player_exploration)
            
            # 行動を実行
            self.env.update(action, self.env.next_player) 

        return self.env.winner()
