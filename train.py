import copy

from env.Reversi import Reversi
from agent.dqn_agent import DQNAgent

from worker.self_play import self_play
from worker.optimize import optimize
from worker.evaluate import evaluate

from datas.MoveHistory import MoveHistory

if __name__ == "__main__":

    env = Reversi()

    # プレイヤーAIを２体用意
    player = dict()
    player[env.Black] = DQNAgent(env.enable_actions, '{}player{}'.format(env.name, env.Black), env.screen_n_rows, env.screen_n_cols)
    player[env.White] = DQNAgent(env.enable_actions, '{}player{}'.format(env.name, env.White), env.screen_n_rows, env.screen_n_cols)

    # コロシアム：自己対戦をする環境
    Colosseum = self_play(env, player)

    # トレイナー：学習を行う環境
    Traner = optimize(env, player)
    
    # エバリューター：評価する環境
    Evaluator = evaluate(env, player)

    game_idx = 0
    while True: # 永久に続ける
        
        # 学習対象プレーヤーを選択する
        target_player = env.Black if game_idx % 2 == 0 else env.White

        # 自己対戦して棋譜を生成する
        move_list = Colosseum.start(target_player, game_idx, False)
        
        # エージェントに棋譜を読み込ませる
        agent = Traner.set_experience(target_player, move_list)
        # 学習データ数を出力
        print("TARGET: {:01d} | learning Data Count {}".format(target_player, len(agent.D))) 

        # n_train 回学習する
        n_train = 1000
        # 勝率
        win_rate = 0.0
        for i in range(n_train):

            # 棋譜から学習する
            Traner.start(target_player, game_idx, agent)

            # 対戦してみる
            win_rate = Evaluator.start(target_player) 

            # 相手AIより強くなったら学習終了
            if win_rate > 0.9:
                 break

        # 相手AIより強くなったら
        if win_rate > 0.9:
            # プレーヤーを保持して
            player[target_player].save_model()
            # ランダムで打つ確率を初期値にセット
            player[target_player].exploration = 0.8
            # 学習する対象AIを変える
            game_idx += 1
        else:
            # n_train 回学習しても相手AIより強くならなかったら
            # もう一回繰り返すが、棋譜生成の際ランダムで打つ確率を上げる
            expl = self.player[target_player].exploration
            player[target_player].exploration = max(expl-0.05,0.4)       

        # 学習の終わった棋譜は削除する
        MoveHistory.remove_play_data(target_player)


