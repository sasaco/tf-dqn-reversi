#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import json
import copy
from tensorflow.python.framework import ops

import numpy as np
import random

class Agent:
    # エージェントが持つ脳となるクラスです。Q学習を実行します。

    def __init__(self, obs_size, n_actions):

        n_nodes = 256 # 中間層のノード数

        self.x = tf.placeholder(tf.float32, [None, obs_size])
        self.y = tf.placeholder(tf.float32, [None, n_actions])
        W1 = tf.Variable(tf.random_normal([obs_size, n_nodes]))
        b1 = tf.Variable(tf.constant(0.1, shape=[n_nodes]))
        h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)
        W2 = tf.Variable(tf.random_normal([n_nodes, n_actions]))
        b2 = tf.Variable(tf.constant(0.1, shape=[n_actions]))
        self.q = tf.nn.relu(tf.matmul(h1, W2) + b2)

        # optimizerの設定
        self.loss = tf.square(self.y - self.q)
        self.train_op = tf.train.AdagradOptimizer(0.01).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        # saver
        self.saver = tf.train.Saver()

        self.n_actions = n_actions
        self.available_index = []

        # 減衰率
        self.gamma = 0.99

        # ε-greedy法
        self.start_epsilon = 1.0
        self.end_epsilon = 0.1
        self.epsilon = self.start_epsilon


        # Expericence Replay用のバッファ（十分大きく）
        self.capacity = 10 ** 6
        self.target_update_interval = 1 * 10**4
        self.replay_start_size = 1000

        # エージェント．DQNを利用．バッチサイズを少し大きめに設定
        self.minibatch_size = 128

        self.replay_buffer = []
        self.last_state = None
        self.last_action = None

        self.step_counter = 0

        self.log_reset()

    def log_reset(self):
         self.log = {'average_q': [], 'average_loss': [], 'n_updates': 0}
       

    def update_Q_table(self, state, available_list, reward, stop=False):

        int_action = -1 
        if self.step_counter == 0:
            int_action = self.agent_start(state, available_list)
        elif stop==False:
            int_action = self.agent_step(reward, state, available_list)
        else:
            self.agent_end(state, reward)
            return

        # state = 盤の状態 と action = ○を打つ場所 を退避する
        self.last_state = copy.deepcopy(state)
        self.last_action = copy.deepcopy(int_action)

        return int_action


    # 1手目の○を決定し、返す
    def agent_start(self, state, available_list):

        self.update_targetQ()

        # stepを1増やす
        self.step_counter += 1

        # ○の場所を決定する
        int_action = self.select_int_action(state, available_list)
 
        # eps を更新する。epsはランダムに○を打つ確率
        self.update_eps()

        return int_action


    # エージェントの二手目以降、ゲームが終わるまで
    def agent_step(self, reward, state, available_list):
        # ステップを1増加
        self.step_counter += 1

        self.update_targetQ()

        # ○の場所を決定
        int_action = self.select_int_action(state, available_list)

        # epsを更新
        self.update_eps()

        # データを保存 (状態、アクション、報酬、結果)
        self.store_transition(state, reward, done=False)

        # 学習実行
        if self.step_counter > self.replay_start_size:
            self.replay_experience()

        # ○の位置をエージェントへ渡す
        return int_action

    # ゲームが終了した時点で呼ばれる
    def agent_end(self, state, reward):

        # データを保存 (状態、アクション、報酬、結果)
        self.store_transition(state, reward, done=True)

        # 学習実行
        if self.step_counter > self.replay_start_size:
            self.replay_experience()


    def update_eps(self):
         # epsを更新 ( ε-greedy法 )
        if self.step_counter > self.replay_start_size:
            if len(self.replay_buffer) < self.capacity:
                self.epsilon -= ((self.start_epsilon - self.end_epsilon) /
                             (self.capacity - self.replay_start_size + 1))


    def update_targetQ(self):
        if self.step_counter % self.target_update_interval == 0:
            self.targetQ = copy.copy(self.q)
            self.log['n_updates'] += 1
 
    def select_int_action(self, state, available_list):

        # Follow the epsilon greedy strategy
        if np.random.rand() < self.epsilon:
            int_action = random.choice(available_list)
        else:
            # self.q がもつ Q値を取得
            Qdata = self.sess.run(self.targetQ, feed_dict={self.x: [state]})[0]

            for i in np.argsort(-Qdata):
                if i in available_list:
                    int_action = i
                    break

        return int_action


    def store_transition(self, state, reward, done=False):
        # データを保存 (状態、アクション、報酬、結果)
        if len(self.replay_buffer) < self.capacity:
            self.replay_buffer.append(
                (self.last_state, self.last_action, reward, state, done))
        else:
            self.replay_buffer = (self.replay_buffer[1:] +
                [(self.last_state, self.last_action, reward, state, done)])


    def replay_experience(self):
        # 学習実行

        indices = np.random.randint(0, len(self.replay_buffer), self.minibatch_size)
        samples = np.asarray(self.replay_buffer)[indices]

        state, action, reward, next_state, done = [], [], [], [], []

        for sample in samples:
            state.append(sample[0])
            action.append(sample[1])
            reward.append(sample[2])
            next_state.append(sample[3])
            done.append(sample[4])

        # self.targetQ がもつ 今の状態の Q値を取得
        Q_values = self.sess.run(self.targetQ, feed_dict={self.x: state})
        # self.targetQ がもつ 次の状態の Q値を取得
        next_Q_values = self.sess.run(self.targetQ, feed_dict={self.x: next_state})

        for i in range(len(Q_values)):

            # 今の状態の Q値 の最大値のインデックスを 取得
            action_index = action[i]

            # 次の状態の Q値 の最大値を 取得
            max_value = np.max(next_Q_values[i])

            # 次の状態の Q値 の最大値を 今の状態の Q値 に反映する
            Q_values[i][action_index] = reward[i] + self.gamma * max_value

        # self.q を loss が小さくなるように self.train_op ルールで更新
        self.sess.run(self.train_op, feed_dict={self.x: state, self.y: Q_values})

        #for log
        self.log['average_loss'].append(self.sess.run(self.loss, feed_dict={self.x: state, self.y: Q_values}))


        self.log['average_q'].append(Q_values)


    def get_statistics(self):

        result = {}
        result['average_q'] = np.average(self.log['average_q'])
        result['average_loss'] = np.average(self.log['average_loss'])
        result['n_updates'] = self.log['n_updates']
        self.log_reset()

        return json.dumps(result)

    def epsilon(self):
        return self.epsilon

    def save(self, dirname):
        save_path = self.saver.save(self.sess, dirname)
