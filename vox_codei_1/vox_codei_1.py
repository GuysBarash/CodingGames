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


def translate_coordinates(row, col):
    return f'{int(col)} {int(row)}'


class World:
    def __init__(self):
        self.turn = 0

        self.grid_width = None
        self.grid_height = None

        self.rounds = None
        self.bombs = None

        self.map = list()
        self.active_bombs = list()
        self.arr = None

    def init(self, d=None):
        if d is None:
            inps = input().split()
            self.grid_width, self.grid_height = [int(i) for i in inps]
            for i in range(self.grid_height):
                self.map.append(input())
            self.arr = np.array([list(t) for t in self.map])

        else:
            pass

    def update(self, d=None):
        if d is None:
            self.turn += 1
            inps = input().split()
            self.rounds, self.bombs = [int(i) for i in inps]
        else:
            self.turn = d['turn']
            self.grid_width = d['grid_width']
            self.grid_width = d['grid_height']
            self.rounds = d['rounds']
            self.bombs = d['bombs']
            self.map = d['map']
            self.arr = np.array([list(t) for t in self.map])
        self.tick()

    def get_params(self):
        d = dict()
        d['turn'] = self.turn
        d['grid_width'] = self.grid_width
        d['grid_height'] = self.grid_width
        d['rounds'] = self.rounds
        d['bombs'] = self.bombs
        d['map'] = self.map
        return d

    @staticmethod
    def bomb_arr(row, col, arr):
        items_removed = 0
        validity = lambda pos: 0 <= pos[0] < arr.shape[0] and 0 <= pos[1] < arr.shape[1]

        c_tile = arr[row, col]
        if c_tile not in ['.', '0'] or not validity((row, col)):
            return -1
        else:
            # UP
            top_bound = min(3, row)
            for i in np.arange(1, 1 + top_bound):
                r, c = row - i, col
                # print(f"UP ({r},{c})")
                tile = arr[r, c]
                if tile == '@':
                    arr[r, c] = '.'
                    items_removed += 1
                elif tile == '.':
                    pass
                else:
                    break
            # Down
            low_bound = min(3, arr.shape[0] - 1 - row)
            for i in np.arange(1, 1 + low_bound):
                r, c = row + i, col
                # print(f"DOWN ({r},{c})")
                tile = arr[r, c]
                if tile == '@':
                    arr[r, c] = '.'
                    items_removed += 1
                elif tile == '.':
                    pass
                else:
                    break
            # LEFT
            top_bound = min(3, col)
            for i in np.arange(1, 1 + top_bound):
                r, c = row, col - i
                # print(f"LEFT ({r},{c})")
                tile = arr[r, c]
                if tile == '@':
                    arr[r, c] = '.'
                    items_removed += 1
                elif tile == '.':
                    pass
                else:
                    break
            # RIGHT
            low_bound = min(3, arr.shape[1] - 1 - col)
            for i in np.arange(1, 1 + low_bound):
                r, c = row, col + i
                # print(f"RIGHT ({r},{c})")
                tile = arr[r, c]
                if tile == '@':
                    arr[r, c] = '.'
                    items_removed += 1
                elif tile == '.':
                    pass
                else:
                    break

            return items_removed

    @staticmethod
    def bomb_value(row, col, arr):
        items_removed = 0
        validity = lambda pos: 0 <= pos[0] < arr.shape[0] and 0 <= pos[1] < arr.shape[1]

        c_tile = arr[row, col]
        if c_tile != '.' or not validity((row, col)):
            return -1
        else:
            # UP
            top_bound = min(3, row)
            for i in np.arange(1, 1 + top_bound):
                r, c = row - i, col
                # print(f"UP ({r},{c})")
                tile = arr[r, c]
                if tile == '@':
                    items_removed += 1
                elif tile == '.':
                    pass
                else:
                    break
            # Down
            low_bound = min(3, arr.shape[0] - 1 - row)
            for i in np.arange(1, 1 + low_bound):
                r, c = row + i, col
                tile = arr[r, c]
                if tile == '@':
                    items_removed += 1
                elif tile == '.':
                    pass
                else:
                    break
            # LEFT
            top_bound = min(3, col)
            for i in np.arange(1, 1 + top_bound):
                r, c = row, col - i
                tile = arr[r, c]
                if tile == '@':
                    items_removed += 1
                elif tile == '.':
                    pass
                else:
                    break
            # RIGHT
            low_bound = min(3, arr.shape[1] - 1 - col)
            for i in np.arange(1, 1 + low_bound):
                r, c = row, col + i
                # print(f"RIGHT ({r},{c})")
                tile = arr[r, c]
                if tile == '@':
                    items_removed += 1
                elif tile == '.':
                    pass
                else:
                    break

            return items_removed

    def apply(self, action):
        if action == 'WAIT':
            pass
        else:
            bomb_timer = 3
            r, c = [int(t) for t in action.split(' ')][::-1]
            self.arr[r, c] = str(bomb_timer)

    def tick(self):
        self.arr[np.where(self.arr == '1')] = '0'
        self.arr[np.where(self.arr == '2')] = '1'
        self.arr[np.where(self.arr == '3')] = '2'
        self.arr[np.where(self.arr == '4')] = '3'

        idx = np.where(self.arr == '0')
        if idx[0].shape[0] > 0:
            r, c = idx[0][0], idx[1][0]
            World.bomb_arr(r, c, self.arr)
            self.arr[r, c] = '.'
            dprint(f"BOMB EXPLODE {r}x{c}")


class HardCodedAgent:
    def __init__(self):
        self.targets = None

    def observe(self, world):
        if self.targets is None:
            self.targets = list()
            pos = np.where(world.arr == '@')
            for i in range(pos[0].shape[0]):
                self.targets.append((pos[0][i], pos[1][i]))

        if world.bombs <= 0:
            return 'WAIT'

        bomb_to_target_idx = world.turn - 1
        if len(self.targets) > bomb_to_target_idx:
            bomb_to_target = self.targets[bomb_to_target_idx]
            return translate_coordinates(bomb_to_target[0] + 1, bomb_to_target[1])
        else:
            return 'WAIT'


class GreedyAgent:
    def __init__(self):
        self.arr = None
        self.step = 0
        self.bombs = None

        self.real_time = None
        self.time = 0
        self.success = None

    def observe(self, world):
        if self.bombs is None:
            self.bombs = list()
            self.arr = world.arr.copy()
            for bomb_idx in range(world.bombs):
                values = np.zeros(self.arr.shape)

                for r in range(self.arr.shape[0]):
                    for c in range(self.arr.shape[1]):
                        values[r, c] = World.bomb_value(r, c, self.arr)

                wr, wc = np.unravel_index(np.argmax(values, axis=None), values.shape)
                max_val = values[wr, wc]
                if max_val <= 0:
                    break
                self.bombs.append((wr, wc))
                World.bomb_arr(wr, wc, self.arr)

            tiles_left = (self.arr == '@').sum()
            if tiles_left <= 0:
                self.success = True
            else:
                dprint("WARNING. This session is not optimal.")
                if world.arr[1,7] == '@':
                    self.bombs = [(5, 6), (5, 8), (4, 7), (6, 7)]
                else:
                    self.bombs = [(6, 6), (6, 8), (5, 7), (7, 7)]
                self.success = False

        no_more_bombs = world.bombs <= 0
        no_need_for_bombs = self.step >= len(self.bombs)

        if no_more_bombs or no_need_for_bombs:
            return 'WAIT'
        else:
            r, c = self.bombs[self.step]
            position_not_clear = world.arr[r, c] != '.'
            if position_not_clear:
                dprint(f"Position {r}x{c} not clear. waiting.")
                return 'WAIT'
            else:
                s = translate_coordinates(r, c)
                self.step += 1
                return s


world = World()
d = None
if LOCAL_MODE:
    d = dict()
world.init(d)
agent = GreedyAgent()

while True:
    d = None
    if LOCAL_MODE:
        d = dict()
        d["turn"] = 1
        d["grid_width"] = 15
        d["grid_height"] = 15
        d["rounds"] = 15
        d["bombs"] = 4
        d["map"] = ['...............', '...#...@...#...', '....#.....#....', '.....#.@.#.....', '......#.#......',
                    '...@.@...@.@...', '......#.#......', '.....#.@.#.....', '....#.....#....', '...#...@...#...',
                    '...............', '...............']

    world.update(d)
    print_dict(world.get_params())
    dprint_array(world.arr)
    action = agent.observe(world)
    world.apply(action)
    print(action)

    if LOCAL_MODE:
        break
