import numpy as np

from Reversi import Reversi
from dqn_agent import DQNAgent

          
if __name__ == "__main__":
    
    # parameters
    n_epochs = 5000
    # environment, agent
    env = Reversi()
    # agent1 = env.Black
    agent1 = DQNAgent(env.enable_actions, env.name, env.screen_n_rows, env.screen_n_cols)

    # agent2 = env.White
    agent2 = DQNAgent(env.enable_actions, env.name, env.screen_n_rows, env.screen_n_cols)

    for e in range(n_epochs):
        # reset
        env.reset()
        terminal = False
        while terminal == False: # 1エピソードが終わるまでループ
 
            """ player = env.Black """
            state0 = env.screen
            targets1 = env.get_enables(env.Black)
            if len(targets1) > 0:
                # どこかに置く場所がある場合  
                action1 = agent1.select_action(state0, targets1, agent1.exploration)
                # 行動を実行
                env.update(action1, env.Black)
                # 行動を実行した結果
                state1 = env.screen
                terminal = env.isEnd()     
                # for log
                loss = agent1.current_loss
                Q_max = np.max(agent1.Q_values(state0))
                print("player:{:1d} | pos:{:2d} | LOSS: {:.4f} | Q_MAX: {:.4f}".format(
                         env.Black, action1, loss, Q_max))
                # 終了してたら報酬1を得る
                if terminal == True:
                    win = env.winner()
                    if win == env.Black:
                       agent1.store_experience(state0, action1, 1, state1, terminal)
                       agent1.experience_replay()
                    elif win == env.White:
                       agent2.store_experience(state0, action1, 1, state1, terminal)
                       agent2.experience_replay()
                    break
                
            """ player = env.White """
            state2 = env.screen
            targets2 = env.get_enables(env.White)
            if len(targets2) > 0:
                # どこかに置く場所がある場合          
                action2 = agent2.select_action(state2, targets2, agent2.exploration)
                # 行動を実行
                env.update(action2, env.White)
                # 行動を実行した結果
                state3 = env.screen
                terminal = env.isEnd() 
                 # for log
                loss = agent2.current_loss
                Q_max = np.max(agent2.Q_values(state2))
                print("player:{:1d} | pos:{:2d} | LOSS: {:.4f} | Q_MAX: {:.4f}".format(
                         env.White, action2, loss, Q_max))
                # 終了してたら報酬1を得る
                if terminal == True:
                    win = env.winner()
                    if win == env.Black:
                       agent1.store_experience(state0, action1, 1, state1, terminal)
                       agent1.experience_replay()
                    elif win == env.White:
                       agent2.store_experience(state2, action2, 1, state3, terminal)
                       agent2.experience_replay()
                    break
                      
            """ 復習（報酬なし） """
            if len(targets1) > 0:
                agent1.store_experience(state0, action1, 0, state1, terminal)
                agent1.experience_replay()
                #対戦相手は、すべての手を登録する
                for tr in targets1:
                    agent2.store_experience(state0, action1, 0, state1, terminal)
                    agent2.experience_replay()
                    
            if len(targets2) > 0:
                agent2.store_experience(state2, action2, 0, state3, terminal)
                agent2.experience_replay()
                #対戦相手は、すべての手を登録する
                for tr in targets2:
                    agent1.store_experience(state2, action2, 0, state3, terminal)
                    agent1.experience_replay()

                                
        w = env.winner()                    
        print("EPOCH: {:03d}/{:03d} | WIN: player{:1d}".format(
                         e, n_epochs, w))


    # 保存は後攻のplayer2 を保存する。
    agent2.save_model()

           
