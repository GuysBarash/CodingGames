import sys
import math
import socket
import re

import numpy as np
import pandas as pd

import pickle as p
from collections import deque
from datetime import datetime
import time
from collections import deque
from queue import PriorityQueue
from copy import deepcopy as copy

DEBUG_MODE = True
LOCAL_MODE = socket.gethostname() == 'Barash-pc'


def dprint(s=''):
    print(s, file=sys.stderr, flush=True)


def dprint_map(m):
    dprint("")
    for mt in m:
        dprint(mt)
    dprint("")


def print_dict(d):
    dprint('d = dict()')
    for k in d.keys():
        v = d[k]
        if type(v) == str:
            dprint(f'd["{k}"] = "{v}"')
        else:
            dprint(f'd["{k}"] = {v}')
    dprint("")


class State:
    def __init__(self, d_start=None, turn=0):
        self.d = d_start

        self.lines = None
        self.map = None
        self.robots = None
        self.robot_count = None
        self.turn = turn
        self.terminal = False
        self.markers = list()

    def export(self):
        # Convert all to dict
        d = dict()
        d['lines'] = self.lines
        d['robots'] = self.robots
        d['robot_count'] = self.robot_count
        d['turn'] = self.turn
        d['terminal'] = self.terminal
        d['score'] = self.score
        d['markers'] = self.markers
        self.d = d

        return d

    def export_and_print(self):
        print_dict(self.export())
        dprint_map(self.map)

    def place_robot(self):
        for r in self.robots:
            if r['terminated']:
                continue
            x, y = r['pos']
            direction = r['dir']
            self.map[y] = self.map[y][:x] + direction + self.map[y][x + 1:]

    def place_marker(self):
        for x, y, direction in self.markers:
            dir_sign = direction.lower()
            if self.map[y][x] == '.':
                self.lines[y] = self.lines[y][:x] + dir_sign + self.lines[y][x + 1:]

    def update(self, d=None):
        if d is None:
            # Get turn parameters from CLI
            self.lines = list()
            for i in range(10):
                line = input().lower()
                self.lines += [line]

            self.robot_count = int(input())
            self.robots = list()
            for i in range(self.robot_count):
                inputs = input().split()
                x = int(inputs[0])
                y = int(inputs[1])
                direction = inputs[2]

                r = dict()
                r['pos'] = x, y
                r['dir'] = direction
                r['idx'] = i
                r['terminated'] = False
                r['hist'] = list()
                self.robots += [r]

            self.terminal = False
            self.score = 0

            # Get position of markers
            self.markers = list()
            for y, line in enumerate(self.lines):
                for x, c in enumerate(line):
                    if c in ['u', 'd', 'r', 'l']:
                        self.markers += [(x, y, c.upper())]
        else:
            # Get turn parameters from d
            self.d = d
            self.lines = d['lines']
            self.robots = d['robots']
            self.robot_count = d['robot_count']
            self.turn = d['turn']
            self.terminal = d['terminal']
            self.markers = d['markers']
            self.score = d['score']

        self.max_y, self.max_x = len(self.lines), len(self.lines[0])

        self.map = copy(self.lines)
        self.place_marker()
        self.place_robot()
        return None

    def apply(self, action):
        action_l = action.split(' ')
        action_count = int(len(action_l) / 3)
        for i in range(action_count):
            x, y, direction = action_l[i * 3:i * 3 + 3]
            x, y = int(x), int(y)
            self.markers += [(x, y, direction)]

    def step(self):
        n_state = State()
        d = self.export()
        d['turn'] += 1
        for robot in d['robots']:
            if robot['terminated']:
                continue
            x, y = robot['pos']
            direction = robot['dir']
            hist_item = f'{x}.{y}.{direction}'
            robot['hist'] += [hist_item]
            if direction == 'U':
                y -= 1
            elif direction == 'D':
                y += 1
            elif direction == 'R':
                x += 1
            elif direction == 'L':
                x -= 1

            y = y % self.max_y
            x = x % self.max_x

            new_sig = f'{x}.{y}.{direction}'
            if new_sig in robot['hist']:
                robot['terminated'] = True
                d['robot_count'] -= 1
                dprint(f'Robot {robot["idx"]} in a loop')
                continue

            # if x < 0 or x >= self.max_x or y < 0 or y >= self.max_y:
            #     robot['terminated'] = True
            #     d['robot_count'] -= 1
            #     dprint(f"Robot {robot['idx']} out of map")
            #     continue

            marker = self.map[y][x]
            if marker == '#':
                robot['terminated'] = True
                d['robot_count'] -= 1
                dprint(f"Robot {robot['idx']} stuck on wall")
                continue

            elif marker == 'u':
                robot['dir'] = 'U'
            elif marker == 'd':
                robot['dir'] = 'D'
            elif marker == 'r':
                robot['dir'] = 'R'
            elif marker == 'l':
                robot['dir'] = 'L'
            else:
                pass

            robot['pos'] = x, y

        if d['robot_count'] <= 0:
            d['terminal'] = True

        d['score'] += d['robot_count']

        n_state.update(d)
        return n_state


class Agent:
    def __init__(self, state=None):
        pass

    def observe(self, state=None):
        action = "14 2 L 8 2 R"
        return action

    def run_action(self, action, state, display=True):
        n_state = copy(state)
        n_state.apply(action)
        while True:
            n_state = n_state.step()
            if display:
                n_state.export_and_print()
                time.sleep(0.3)
            if n_state.terminal:
                if display:
                    n_state.export_and_print()
                break

        expeted_score = n_state.score
        possible_moves = self.get_possible_markers(n_state)
        dprint(f'Expected score: {expeted_score}')
        print(action)

    def get_possible_markers(self, state):
        markers = list()
        hists = [r['hist'] for r in state.robots]
        flat_hists = [item for sublist in hists for item in sublist]
        hist_coords = [tuple(map(int, h.split('.')[0:2])) for h in flat_hists]
        hist_coords = list(set(hist_coords))

        # Remove occupied coords
        occupied_coords = [(x, y) for x, y, _ in state.markers]
        hist_coords = [c for c in hist_coords if c not in occupied_coords]

        dirs = ['U', 'D', 'R', 'L']

        possibles = list()
        for dirc in dirs:
            possibles += [(x, y, dirc) for (x, y) in hist_coords]


d = None
if LOCAL_MODE:
    d = None

state = State(d)
agent = Agent()

section_main = True
if section_main:
    if LOCAL_MODE:
        d = dict()
        d["lines"] = ['l..#############..u', '...#############...', '...#############...', '###################',
                      '###################', '###################', '###################', '...#############...',
                      '...#############...', 'd..#############..r']
        d["robots"] = [{'pos': (2, 2), 'dir': 'L', 'idx': 0, 'terminated': False, 'hist': []},
                       {'pos': (16, 2), 'dir': 'U', 'idx': 1, 'terminated': False, 'hist': []},
                       {'pos': (2, 7), 'dir': 'D', 'idx': 2, 'terminated': False, 'hist': []},
                       {'pos': (16, 7), 'dir': 'R', 'idx': 3, 'terminated': False, 'hist': []}]
        d["robot_count"] = 4
        d["turn"] = 0
        d["terminal"] = False
        d["score"] = 0
        d["markers"] = [(0, 0, 'L'), (18, 0, 'U'), (0, 9, 'D'), (18, 9, 'R'), (14, 2, 'L'), (8, 2, 'R')]
    else:
        pass

    state.update(d)
    action = agent.observe(state)
    agent.run_action(action, state, display=False)

    state.apply(action)
    state.export_and_print()
    print(action)

    if LOCAL_MODE:
        while True:
            state = state.step()
            state.export_and_print()
            time.sleep(0.3)
            if state.terminal:
                state.export_and_print()
                break
