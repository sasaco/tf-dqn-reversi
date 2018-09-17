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

    # プレーヤーの学習前の勝率
    player_win_rate = dict()
    player_win_rate[env.Black] = 0.0
    player_win_rate[env.White] = 0.0

    # コロシアム：自己対戦をする環境
    Colosseum = self_play(env, player)

    # トレイナー：学習を行う環境
    Traner = optimize(env, player)
    
    # エバリューター：評価する環境
    Evaluator = evaluate(env, player)

    i = 0
    while True: # 永久に続ける
        
        # 学習対象プレーヤーを選択する
        target_player = env.Black if i % 2 == 0 else env.White

        # 学習前のプレーヤーを保持しておく
        player[target_player].save_model()

        # self:自己対戦して棋譜を保存する
        Colosseum.start(target_player)
            
        # opt:棋譜から学習する
        loss = Traner.start(target_player)

        # 学習の終わった棋譜は削除する
        MoveHistory.remove_play_data(target_player)

        # eval:対戦して勝率を
        win_rate = Evaluator.start(target_player)

        # 前のAIより強くなったのか判定
        if win_rate < player_win_rate[target_player]:
            # 弱かったら元に戻す
            player[target_player].load_model()
            # ランダムに打つ確率を増やす
            player[target_player].exploration = win_rate
        else:
            # ランダムに打つ確率を減らす
            player[target_player].exploration = win_rate
            # 次のプレーヤーに交代
            i += 1


