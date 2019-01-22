#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 22:39:53 2019

@author: zhaoyu
"""

import TD as td
from maze import Maze

class ReinLrn(object):
    def __init__(self, rl_func, actions, nrow=4, ncol=4):
        self.env = Maze(nrow, ncol)
        self.actions = actions
        self.tab = rl_func(self.env.state_num(), len(self.actions))

    def update(self):
        pass

    def run(self):
        self.update()
        # self.env.after(0, self.update)
        self.env.mainloop()

class ReinLrnQ(ReinLrn, object):
    def __init__(self, actions, nrow=4, ncol=4):
        super(ReinLrnQ, self).__init__(td.QLearning, actions, nrow, ncol)

    def update(self):
        for episode in range(100):
            state_0 = self.env.reset()
            while True:
                self.env.render()
                action = self.tab.choose_action(state_0)
                state_1, r, done = self.env.step(action)
                # online-learning strategy
                self.tab.learn(state_0, action, r, state_1)
                if done == True:
                    break
                state_0  = state_1
        print('One-pass Iteration Finished')

class ReinLrnS(ReinLrn, object):
    def __init__(self, actions, nrow=4, ncol=4):
        super(ReinLrnS, self).__init__(td.Sarsa, actions, nrow, ncol)

    def update(self):
        for episode in range(100):
            state_0 = self.env.reset()
            action_0 = self.tab.choose_action(state_0)
            while True:
                self.env.render()
                state_1, r, done = self.env.step(action_0)
                action_1 = self.tab.choose_action(state_1)
                # online-learning strategy
                self.tab.learn(state_0, action_0, r, state_1, action_1)  # The Origin of the Name of Sarsa
                state_0, action_0 = state_1, action_1
                if done == True:
                    break
        print('One-pass Iteration Finished')

if __name__ == "__main__":
    actions = ['u', 'd', 'r', 'l']
    demo = ReinLrnS(actions)
    demo.run()