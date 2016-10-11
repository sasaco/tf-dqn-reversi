import sys
import numpy as np

from Reversi import Reversi
from dqn_agent import DQNAgent


if __name__ == "__main__":
    # parameters
    n_epochs = 5000
    
    # environment, agent
    env = Reversi()
    players = []
    # player 1 = env.Black
    players.append(DQNAgent(env.enable_actions, env.name, env.screen_n_rows, env.screen_n_cols))
    # player 2 = env.White
    players.append(DQNAgent(env.enable_actions, env.name, env.screen_n_rows, env.screen_n_cols))

    # variables
    win = 0

    for e in range(n_epochs):
        # reset
        loss = [0, 0]
        Q_max = [0, 0]
        env.reset()
        state_t_1, reward_t, terminal = env.observe()

        while not terminal: # 1エピソードが終わるまでループ
            state_t = state_t_1
            
            for i in range(0, len(players)):
                if env.isEnd() == True:
                    terminal = True
                    break

                agent = players[i]
                enable = 0
                
                targets = env.get_enables(i+1)
                if len(targets) > 0:
                    # どこかに置く場所がある場合
                    while not enable > 0: #有効な選択がなされるまで繰り返す
                        # 行動を選択
                        if len(targets) == 1:
                            # 1箇所しか置く場所がない
                            action_t = targets[0]
                            print("player:{:1d} is only pos:{:2d}".format(i+1, action_t))
                        else:
                            action_t = agent.select_action(state_t, targets, agent.exploration)
                            
                        # 行動を実行
                        enable = env.execute_action(action_t, i+1)
                        # 報酬を得る
                        state_t_1, reward_t, terminal = env.observe()
                        # store experience
                        agent.store_experience(state_t, action_t, reward_t, state_t_1, terminal)
                        # experience replay
                        agent.experience_replay()
                    
                    # for log
                    loss[i] += agent.current_loss
                    Q_max[i] += np.max(agent.Q_values(state_t))
                    print("player:{:1d} | pos:{:2d} | LOSS: {:.4f} | Q_MAX: {:.4f}".format(
                        i+1, action_t, loss[i], Q_max[i]))
                
                
                elif env.isEnd() == True:
                    #双方置けなくなったらゲーム終了
                    print("player1, 2 is pass")
                    env.print_screen()
                    terminal = True
                    break 
                else:
                    # どこにも置く場所がない
                    print("player:{:1d} is pass".format(i+1))
                  
            w = env.winner()
            if w == env.Black:
                win += 1               
            elif w == env.White:
                win -= 1               
                        
        print("EPOCH: {:03d}/{:03d} | WIN: player{:1d} | LOSS: {:.4f} | Q_MAX: {:.4f}".format(
            e, n_epochs - 1, w+1, loss[w-1], Q_max[w-1]))

    # save model
    if win > 0:
        print('winner is player{:}'.format(1))
        # players[0].save_model()
    else:
        print('winner is player{:}'.format(2))
        # players[1].save_model()

    # 保存は後攻のplayer2 を保存する。
    players[1].save_model()
