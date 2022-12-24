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
import itertools

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

    def export_and_print(self, dict_mode=False):
        if dict_mode:
            d = self.export()
            print_dict(d)
        else:
            d = self.export()
            dx = {k: v for k, v in d.items() if k not in ['robots']}
            robots = d['robots']
            for r in robots:
                msg = ''
                rx = {k: v for k, v in r.items() if k not in ['hist']}
                msg += f"{rx}\n"
                msg += f"{r['hist']}"
                dprint(msg)
            print_dict(dx)
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
                r['hist'] = [f'{x}.{y}.{direction}']
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
        if action != None:
            action_l = action.split(' ')
            action_count = int(len(action_l) / 3)
            for i in range(action_count):
                x, y, direction = action_l[i * 3:i * 3 + 3]
                x, y = int(x), int(y)
                self.markers += [(x, y, direction)]

        # Check if there is a robot on the marker
        for x, y, direction in self.markers:
            for r in self.robots:
                rx, ry = r['pos']
                if rx == x and ry == y:
                    r['dir'] = direction
        j = 3

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
                # dprint(f'Robot {robot["idx"]} in a loop')
                continue
            else:
                robot['hist'] += [new_sig]

            marker = self.map[y][x]
            if marker == '#':
                robot['terminated'] = True
                d['robot_count'] -= 1
                # dprint(f"Robot {robot['idx']} stuck on wall")
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
        curr_state = copy(state)
        moves = list()
        round = -1
        while True:
            round += 1
            cur_run = self.run_action(state=curr_state)
            cur_score = cur_run.score
            possible_actions = self.get_possible_actions(state=cur_run)
            possible_actions_str = [f'{a[0]} {a[1]} {a[2]}' for a in possible_actions]
            possible_actions_str += ['']

            psr = itertools.combinations(possible_actions_str, 2)
            possible_actions_str = [' '.join(t) for t in psr]
            actions = dict()
            for action in possible_actions_str:
                a_run = self.run_action(state=curr_state, action=action, display=False)
                a_score = a_run.score
                actions[action] = a_score

            best_action = max(actions, key=actions.get)
            expected_score = actions[best_action]
            if expected_score > cur_score:
                moves += [best_action]
                curr_state.apply(action=best_action)
                dprint(f'Round {round} - {best_action} - {expected_score}')
            else:
                dprint(f'Best action: NOTHING')
                dprint(f'Expected score: {cur_score}')
                break

        return ' '.join(moves)

    def run_action(self, state, action=None, display=False):
        n_state = copy(state)
        if action is not None:
            n_state.apply(action)
        while True:
            n_state = n_state.step()
            if display:
                n_state.export_and_print()
                # time.sleep(0.3)
            if n_state.terminal:
                if display:
                    n_state.export_and_print()
                break

        return n_state

    def get_possible_actions(self, state):
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

        return possibles


d = None
if LOCAL_MODE:
    d = None

state = State(d)
agent = Agent()

section_main = True
if section_main:
    if LOCAL_MODE:
        d = dict()
        d["lines"] = ['#........l........#', '#........r........#', '#........l........#', '#........r........#',
                      '#dudududu#udududud#', '#........r........#', '#........l........#', '#........r........#',
                      '#........l........#', '###################']
        d["robots"] = [{'pos': (1, 0), 'dir': 'R', 'idx': 0, 'terminated': False, 'hist': ['1.0.R']},
                       {'pos': (17, 0), 'dir': 'D', 'idx': 1, 'terminated': False, 'hist': ['17.0.D']},
                       {'pos': (1, 8), 'dir': 'U', 'idx': 2, 'terminated': False, 'hist': ['1.8.U']},
                       {'pos': (17, 8), 'dir': 'L', 'idx': 3, 'terminated': False, 'hist': ['17.8.L']}]
        d["robot_count"] = 4
        d["turn"] = 0
        d["terminal"] = False
        d["score"] = 0
        d["markers"] = [(9, 0, 'L'), (9, 1, 'R'), (9, 2, 'L'), (9, 3, 'R'), (1, 4, 'D'), (2, 4, 'U'), (3, 4, 'D'),
                        (4, 4, 'U'), (5, 4, 'D'), (6, 4, 'U'), (7, 4, 'D'), (8, 4, 'U'), (10, 4, 'U'), (11, 4, 'D'),
                        (12, 4, 'U'), (13, 4, 'D'), (14, 4, 'U'), (15, 4, 'D'), (16, 4, 'U'), (17, 4, 'D'), (9, 5, 'R'),
                        (9, 6, 'L'), (9, 7, 'R'), (9, 8, 'L')]


    else:
        pass

    state.update(d)
    state.export_and_print(dict_mode=True)
    action = agent.observe(state)
    # agent.run_action(action, state, display=False)

    state.apply(action)
    state.export_and_print()
    print(action)

    if LOCAL_MODE:
        while True:
            state = state.step()
            state.export_and_print()
            if state.terminal:
                state.export_and_print()
                break
