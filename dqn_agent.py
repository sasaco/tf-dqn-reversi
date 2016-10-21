from collections import deque
import os
import sys

import numpy as np
import tensorflow as tf


class DQNAgent:
    """
    Multi Layer Perceptron with Experience Replay
    """

    def __init__(self, enable_actions, environment_name, rows, cols):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.environment_name = environment_name
        self.enable_actions = enable_actions.tolist()
        self.n_actions = len(self.enable_actions)
        self.rows = rows
        self.cols = cols
        self.minibatch_size = 32
        self.replay_memory_size = 1000
        self.learning_rate = 0.001
        self.discount_factor = 0.9
        self.exploration = 0.1
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.model_name = "{}.ckpt".format(self.environment_name)

        # replay memory
        self.D = deque(maxlen=self.replay_memory_size)

        # model
        self.init_model()

        # variables
        self.current_loss = 0.0

    def init_model(self):
        # input layer (rows x cols)
        self.x = tf.placeholder(tf.float32, [None, self.rows, self.cols])

        # flatten (rows x cols)
        size = self.rows * self.cols
        x_flat = tf.reshape(self.x, [-1, size])

        # fully connected layer (32)
        #W_fc1 = tf.Variable(tf.random_uniform(shape=[size, self.n_actions], minval=0,maxval=1,dtype=tf.float32))
        W_fc1 = tf.Variable(tf.truncated_normal([size, size], stddev=0.01))
        b_fc1 = tf.Variable(tf.zeros([size]))
        h_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)

        # output layer (n_actions)
        #W_out = tf.Variable(tf.random_uniform(shape=[size, self.n_actions], minval=0,maxval=1,dtype=tf.float32))
        W_out = tf.Variable(tf.truncated_normal([size, self.n_actions], stddev=0.01))
        b_out = tf.Variable(tf.zeros([self.n_actions]))
        
        self.y = tf.matmul(h_fc1, W_out) + b_out
        
        # loss function
        self.y_ = tf.placeholder(tf.float32, [None, self.n_actions])
        self.loss = tf.reduce_mean(tf.square(self.y_ - self.y))

        # train operation
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.training = optimizer.minimize(self.loss)

        # saver
        self.saver = tf.train.Saver()

        # session
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def Q_values(self, state):
        # Q(state, action) of all actions
        Qs = self.sess.run(self.y, feed_dict={self.x: [state]})[0]
        return Qs

    def select_action(self, state, targets, epsilon):
        
        while True:
            act = self.enable_actions[np.argmax(self.Q_values(state))]
            if act in targets:
                break
            else:
                """ the choice not effective then learning to understand """
                reward_t = -1
                self.store_experience(state, act, reward_t, state, False)
                self.experience_replay()

                
        if np.random.rand() <= epsilon:
            # random
            return np.random.choice(targets)
        else:
            # max_action Q(state, action)
            return act

    def store_experience(self, state, action, reward, state_1, terminal):
        self.D.append((state, action, reward, state_1, terminal))

    def experience_replay(self):
        state_minibatch = []
        y_minibatch = []

        # sample random minibatch
        minibatch_size = min(len(self.D), self.minibatch_size)
        minibatch_indexes = np.random.randint(0, len(self.D), minibatch_size)

        for j in minibatch_indexes:
            state_j, action_j, reward_j, state_j_1, terminal = self.D[j]
            action_j_index = self.enable_actions.index(action_j)

            y_j = self.Q_values(state_j)

            if terminal:
                y_j[action_j_index] = reward_j
            else:
                # reward_j + gamma * max_action' Q(state', action')
                y_j[action_j_index] = reward_j + self.discount_factor * np.max(self.Q_values(state_j_1))

            state_minibatch.append(state_j)
            y_minibatch.append(y_j)

        # training
        self.sess.run(self.training, feed_dict={self.x: state_minibatch, self.y_: y_minibatch})

        # for log
        self.current_loss = self.sess.run(self.loss, feed_dict={self.x: state_minibatch, self.y_: y_minibatch})

    def load_model(self, model_path=None):
        if model_path:
            # load from model_path
            self.saver.restore(self.sess, model_path)
        else:
            # load from checkpoint
            checkpoint = tf.train.get_checkpoint_state(self.model_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

    def save_model(self):
        self.saver.save(self.sess, os.path.join(self.model_dir, self.model_name))
