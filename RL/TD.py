#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 14:31:34 2019

@author: zhaoyu
"""

import numpy as np

class QLearning():
    def __init__(self, n_states, n_actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.state_num, self.action_num = n_states, n_actions
        self.tab = np.zeros([self.state_num, self.action_num], dtype=np.float64)
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

    def learn(self, s, a, r, s_):
        q_predict = self.tab[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.tab[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.tab[s, a] += self.lr * (q_target - q_predict)  # update

    def choose_action(self, state):
        if state == 'terminal':
            return None
        if np.random.uniform(0, 1) < self.epsilon:
            action_list = self.tab[state, :] # choose best action
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(np.where(action_list == np.max(action_list))[0])
        else:
            action = np.random.choice(list(range(self.action_num))) # choose random action
        return action

class Sarsa(QLearning, object):
    def __init__(self, n_states, n_actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(Sarsa, self).__init__(n_states, n_actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        q_predict = self.tab[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.tab[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.tab[s, a] += self.lr * (q_target - q_predict)  # update