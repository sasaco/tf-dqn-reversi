#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
import random


### Q関数の定義 ###
class Brain(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_nodes):
        w = chainer.initializers.HeNormal(scale=0.1) # 重みの初期化
        super(Brain, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(obs_size, n_nodes, initialW=w)
            self.l2 = L.Linear(n_nodes, n_nodes, initialW=w)
            self.l3 = L.Linear(n_nodes, n_nodes, initialW=w)
            self.l4 = L.Linear(n_nodes, n_actions, initialW=w)

    def __call__(self, x):
        h = F.sigmoid(self.l1(x))
        h = F.sigmoid(self.l2(h))
        h = F.sigmoid(self.l3(h))        
        return chainerrl.action_value.DiscreteActionValue(self.l4(h))


class Agent:
    # エージェントが持つ脳となるクラスです。Q学習を実行します。
 
    def __init__(self, obs_size, n_actions):

        n_nodes = 256 # 中間層のノード数
        q_func = Brain(obs_size, n_actions, n_nodes)
        self.n_actions = n_actions
        self.available_index = []

        # optimizerの設定
        optimizer = chainer.optimizers.Adam(eps=1e-3)
        optimizer.setup(q_func)

        # 減衰率
        gamma = 0.99

        # ε-greedy法
        explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=1.0, end_epsilon=0.1, decay_steps=50000, random_action_func=self.random_action)

        # Expericence Replay用のバッファ（十分大きく）
        replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

        # エージェント．DQNを利用．バッチサイズを少し大きめに設定
        self.agent = chainerrl.agents.DQN(
            q_func, optimizer, replay_buffer, gamma, explorer, 
            replay_start_size=1000, minibatch_size=128, update_interval=1, target_update_interval=1000)
 
        
    def update_Q_table(self, state, available_list, reward, stop=False):

        # 配置可能な位置を記録
        self.available_index = available_list

        if stop == False:
            # QテーブルをQ学習により更新
            return self.agent.act_and_train(state, reward)
        else:
            self.agent.stop_episode_and_train(state, reward, True)


    # ランダムに石を置く場所を決める（ε-greedy用）
    def random_action(self):
        return random.choice(self.available_index) # 置く場所をランダムに決める

    def get_statistics(self):
        return self.agent.get_statistics()

    def epsilon(self):
        return self.agent.explorer.epsilon

    def save(self, dirname):
        self.agent.save(dirname)

    def load(self, dirname):
        self.agent.load(dirname)

    def act(self, state, available_index):

        int_action = self.agent.act(state)
        return int_action
