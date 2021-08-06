import sys
import math
import socket

import numpy as np
import pandas as pd

import pickle as p
from copy import copy
from collections import deque
from datetime import datetime
import time

DEBUG_MODE = True
LOCAL_MODE = socket.gethostname() == 'ILB001119'


def dprint(s=''):
    print(s, file=sys.stderr)


def print_dict(d):
    dprint('')
    dprint('d = dict()')
    for k in d.keys():
        v = d[k]
        if type(v) == str:
            dprint(f'd["{k}"] = "{v}"')
        else:
            dprint(f'd["{k}"] = {v}')
    dprint("")


def dprint_array(arr):
    for ridx in range(arr.shape[0]):
        dprint(''.join(arr[ridx]))


class World:
    def __init__(self):
        self.row_count = None
        self.rows = None
        self.arr = None

    def visualize(self):
        pass

    def update(self, d=None):
        read_time = datetime.now()
        if d is None:
            self.row_count = int(input())
            self.rows = dict()
            for i in range(self.row_count):
                self.rows[i] = input()
            dprint("Update completed.")
        else:
            self.row_count = d['row_count']
            self.rows = d['rows']

        self.arr = np.array([[list(self.rows[i]) for i in range(self.row_count)]])
        read_time = (datetime.now() - read_time).total_seconds()
        dprint(f"Read time: {read_time}")

    def get_params(self):
        d = dict()
        d['row_count'] = self.row_count
        d['rows'] = self.rows
        return d


class Agent:
    # DIM 1: depth
    # DIM 2: up down
    # DIM 3: Left right

    def __init__(self, world):
        self.world = world

    def observe(self):
        self.X = self.world.arr
        while True:
            if self.check_terminal():
                break

            self.X = self.fold_right2left(self.X)
            if self.check_terminal():
                break

            self.X = self.fold_bottom2top(self.X)
            if self.check_terminal():
                break

            self.X = self.fold_left2right(self.X)
            if self.check_terminal():
                break

            self.X = self.fold_top2bottom(self.X)
            if self.check_terminal():
                break

        # Parse results
        ret = ''.join([self.X[i, 0, 0] for i in range(self.X.shape[0])])
        return ret

    def fold_right2left(self, X):
        mark = int(X.shape[2] / 2)
        left = X[:, :, :mark]
        right = X[:, :, mark:]
        right = np.flip(np.flip(right, axis=0), axis=2)
        Q = np.concatenate([right, left], axis=0)
        return Q

    def fold_left2right(self, X):
        mark = int(X.shape[2] / 2)
        left = X[:, :, :mark]
        right = X[:, :, mark:]
        left = np.flip(np.flip(left, axis=0), axis=2)
        Q = np.concatenate([left, right], axis=0)
        return Q

    def fold_bottom2top(self, X):
        mark = int(X.shape[1] / 2)
        top = X[:, :mark, :]
        bottom = X[:, mark:, :]
        bottom = np.flip(np.flip(bottom, axis=0), axis=1)
        Q = np.concatenate([bottom, top], axis=0)
        return Q

    def fold_top2bottom(self, X):
        mark = int(X.shape[1] / 2)
        top = X[:, :mark, :]
        bottom = X[:, mark:, :]
        top = np.flip(np.flip(top, axis=0), axis=1)
        Q = np.concatenate([top, bottom], axis=0)
        return Q

    def check_terminal(self):
        s = self.X.shape
        return (s[1]) == 1 and (s[2] == 1)


d = None
if LOCAL_MODE:
    d = dict()
    d["row_count"] = 4
    d["rows"] = {0: 'uDuu', 1: 'u!eu', 2: 'uudu', 3: 'uuuu'}

world = World()
world.update(d)
if LOCAL_MODE:
    # world.visualize()
    pass

print_dict(world.get_params())
agent = Agent(world)
res = agent.observe()
time.sleep(0.01)
print(res)
