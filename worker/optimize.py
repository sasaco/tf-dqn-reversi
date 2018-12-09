
from agent.dqn_agent import DQNAgent
from datas.MoveHistory import MoveHistory

class optimize:

    def __init__(self, env, player: dict):
        self.env = env
        self.player = player

    # エージェントに棋譜を読み込ませる
    def set_experience(self, learning_target_player: int, move_list = None):

        # 自己対戦データを読み込む
        if move_list == None:
            move_list = MoveHistory.load_play_data(learning_target_player)

        # 学習対象のAI
        agent = self.player[learning_target_player]

        # 学習データを初期化
        agent.restore_experience()

        # 学習データを追加
        for move in move_list:
            agent.store_experience(move.state, 
                                   move.targets,
                                   move.action, 
                                   move.reward, 
                                   move.state_1, 
                                   move.targets_1, 
                                   move.terminal)

        return agent


    # 棋譜から学習する
    def start(self, learning_target_player: int, game_idx: int, agent: DQNAgent):

        # 学習開始
        agent.experience_replay()  
        # 最適解とのカイ離量
        loss = agent.current_loss

        # 学習回数 | 学習対象プレーヤー | optimize | LOSS
        print("{} | TARGET: {:01d} | optimize | LOSS: {:.6f}".format(
                            game_idx,
                            learning_target_player,
                            loss))

        return loss

