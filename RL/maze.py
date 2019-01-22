"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the environment part of this example. The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
The reward are defined in the environment.
"""

import numpy as np
import time
import tkinter as tk

SIDE = 100   # pixels

class Maze(tk.Tk, object):
    def __init__(self, nrow=4, ncol=4, title='Maze'):
        super(Maze, self).__init__()
        self.blk_sz, self.nrow, self.ncol = SIDE, nrow, ncol
        self.height = self.nrow*self.blk_sz
        self.width = self.ncol*self.blk_sz
        self.action_space = ['u', 'd', 'l', 'r']
        self.delay = 0.1
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(self.height, self.width))
        self._build_maze()

    def _draw_blk(self, r, c, color):
        origin = np.array([self.blk_sz*r, self.blk_sz*c])
        return self.canvas.create_rectangle(
            origin[0], origin[1],
            origin[0]+self.blk_sz, origin[1]+self.blk_sz,
            fill=color)
        
    def _draw_oval(self, r, c, color):
        origin = np.array([self.blk_sz*r, self.blk_sz*c])
        return self.canvas.create_oval(
            origin[0], origin[1],
            origin[0]+self.blk_sz, origin[1]+self.blk_sz,
            fill=color)

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=self.height,
                           width=self.width)

        # create grids
        for c in range(0, self.height, self.blk_sz):
            x0, y0, x1, y1 = c, 0, c, self.height
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.width, self.blk_sz):
            x0, y0, x1, y1 = 0, r, self.width, r
            self.canvas.create_line(x0, y0, x1, y1)

        # hell
        self.hell1 = self._draw_blk(2, 1, 'black')
        self.hell2 = self._draw_blk(1, 2, 'black')

        # create oval
        self.oval = self._draw_oval(2, 2, 'yellow')

        # create red rect
        self.rect = self._draw_blk(0, 0, 'red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(self.delay)
        self.canvas.delete(self.rect)
        self.rect = self._draw_blk(0, 0, 'red')
        return self.get_state(self.rect)

    def step(self, action):
        s = self.get_pos(self.rect)
        base_action = np.array([0, 0])
        if action == 0 and s[1] != 0:
                base_action[1] -= self.blk_sz
        elif action == 1 and s[1] != self.nrow-1:
                base_action[1] += self.blk_sz
        elif action == 2 and s[0] != self.ncol-1:
                base_action[0] += self.blk_sz
        elif action == 3 and s[0] != 0:
                base_action[0] -= self.blk_sz

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent
        s_ = self.get_state(self.rect)

        # reward function
        if s_ == self.get_state(self.oval):
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in [self.get_state(self.hell1), self.get_state(self.hell2)]:
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False

        return s_, reward, done
    
    def render(self):
        time.sleep(self.delay)
        self.update()
    
    def random_walk(self, length=10):
        for it in range(length):
            print(self.get_pos(self.rect))
            a = np.random.randint(0, 4)
            s, r, done = maze.step(a)
            self.render()
            if done:
                break
        print('finish random walk')
            
    def run(self, cnt=3):
        for it in range(cnt):
            print('run from the beginning')
            self.reset()
            self.random_walk(5)

    def run_v2(self):
        print('run from the beginning')
        self.reset()
        self.random_walk(10)
        self.after(0, self.run_v2)

    def get_pos_explorer(self):
        return self.get_pos(self.rect)

    def get_pos(self, tgt):
        w, h = self.canvas.coords(tgt)[:2]
        return np.array([int(w / self.blk_sz), int(h / self.blk_sz)])

    def get_state(self, tgt):
        c, r = self.get_pos(tgt)
        return r*self.ncol + c

    def state_num(self):
        return self.nrow * self.ncol


if __name__ == '__main__':
    maze = Maze()
    maze.run_v2()
    maze.mainloop()