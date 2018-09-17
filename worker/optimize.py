

from datas.MoveHistory import MoveHistory

class optimize:

    def __init__(self, env, player: dict):
        self.env = env
        self.player = player

    def set_experience(self, learning_target_player: int, move_list = None):

        # 自己対戦データを読み込む
        if move_list == None:
            move_list = MoveHistory.load_play_data(learning_target_player)

        # 学習対象
        agent = self.player[learning_target_player]

        # 学習データを初期化
        agent.restore_experience()

        # 学習データを追加
        for move in move_list:
            state = move.state
            targets = move.targets
            action = move.action
            reword = move.reward
            state_X = move.state_1
            target_X = move.targets_1
            end = move.terminal
            agent.store_experience(move.state, targets, action, reword, state_X, target_X, end)

        return agent

    def start(self, learning_target_player: int, agent):

        # 学習開始
        agent.experience_replay()  
        loss = agent.current_loss

        print("TARGET: {:01d} | optimize | LOSS: {:.6f}".format(
                            learning_target_player, loss))

        return loss

