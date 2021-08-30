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

    def __init__(self, d=None):
        if d is None:
            self.board_size = 7, 8  # Y, X
            self.turn = 0
            self.my_id, self.opp_id = [int(i) for i in input().split()]
        else:
            self.board_size = d['board_size']
            self.turn = d['turn']
            self.my_id, self.opp_id = d['my_id'], d['opp_id']

        self.sign_translation = {str(self.my_id): +1, str(self.opp_id): -1, '.': 0}
        self.arr = np.zeros(self.board_size, dtype=int)
        self.state = None

    def get_init_params(self):
        d = dict()
        d['board_size'] = self.board_size
        d['turn'] = self.turn
        d['my_id'] = self.my_id
        d['opp_id'] = self.opp_id
        return d

    def visualize(self):
        pass

    def update(self, d=None):
        read_time = datetime.now()
        if d is None:
            # starts from 0; As the game progresses, first player gets [0,2,4,...] and second player gets [1,3,5,...]
            self.turn = int(input())
            self.raw = [input() for _ in range(7)]
            self.num_valid_actions = int(input())  # number of unfilled columns in the board
            self.valid_actions = [int(input()) for _ in range(self.num_valid_actions)]
            self.current_player = +1

            # opponent's previous chosen column index (will be -1 for first player in the first turn)
            self.opp_previous_action = int(input())

        else:
            self.raw = d['raw']
            self.turn = d['turn']
            self.opp_previous_action = d['opp_previous_action']
            self.num_valid_actions = d['num_valid_actions']
            self.valid_actions = d['valid_actions']
            self.current_player = d['current_player']

        self.arr = np.array([list(map(self.sign_translation.get, list(t))) for t in self.raw])
        self.state = State(self.arr, self.turn, self.current_player, self.opp_previous_action)
        read_time = (datetime.now() - read_time).total_seconds()
        dprint(f"Read time: {read_time}")

    def get_params(self):
        d = dict()
        d['raw'] = self.raw
        d['opp_previous_action'] = self.opp_previous_action
        d['num_valid_actions'] = self.num_valid_actions
        d['valid_actions'] = self.valid_actions
        d['turn'] = self.turn
        d['current_player'] = self.current_player
        return d


class State:
    def __init__(self, arr=None, turn=0, current_player=1, last_move=-2):
        self.last_move = last_move
        self.arr = arr
        self.turn = turn
        self.current_player = current_player

    def apply(self, pos):
        row = np.where(self.arr[:, pos] == 0)[0][-1]
        tarr = self.arr.copy()
        tarr[row, pos] = self.current_player
        s = State(tarr, self.turn + 1, -1 * self.current_player, pos)
        return s

    def isTerminal(self):
        if self.last_move < 0:
            return None

        # 0 for end of board
        # 1 for 1 player
        # -1 for -1 player
        # None for no winner
        # TODO

        pass


class Agent:
    def __init__(self, world):
        pass

    def observe(self, world):
        state = world.state
        action = np.random.choice(world.valid_actions)
        state.apply(action)
        return action


d = None
if LOCAL_MODE:
    d = dict()
    d["board_size"] = (7, 8)
    d["turn"] = 0
    d["my_id"] = 0
    d["opp_id"] = 1

world = World(d)
print_dict(world.get_init_params())
agent = Agent(world)

while True:
    d = None
    if LOCAL_MODE:
        d = dict()
        d["raw"] = ['.........', '.........', '.........', '1........', '0........', '1.1......', '0001.....']
        d["opp_previous_action"] = 3
        d["num_valid_actions"] = 9
        d["valid_actions"] = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        d["turn"] = 8
        d["current_player"] = 1
    world.update(d)
    print_dict(world.get_params())

    res = agent.observe(world)
    print(res)

    if LOCAL_MODE:
        dprint("Locally breaking...")
        break
