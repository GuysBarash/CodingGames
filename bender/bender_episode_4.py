import sys
import math
import socket
import re
from collections import defaultdict

import numpy as np
import pandas as pd

import pickle as p
from copy import deepcopy
from copy import copy
from collections import deque
from datetime import datetime
import time
import queue
from queue import PriorityQueue

DEBUG_MODE = True
LOCAL_MODE = socket.gethostname() == 'Barash-pc'

global lines
global stime
lines = None
boolmap = {True: '1', False: '0'}


def dprint(s=''):
    print(s, file=sys.stderr, flush=True)


def print_dict(d):
    dprint('d = dict()')
    for k, v in d.items():
        if type(v) == str:
            dprint(f'd["{k}"] = "{v}"')
        else:
            dprint(f'd["{k}"] = {v}')
    dprint("")
    m = deepcopy(lines)

    # Add Bender (A)
    x, y = d['start_x'], d['start_y']
    m[y] = m[y] = m[y][:x] + 'A' + m[y][x + 1:]

    # Add Fry (B)
    x, y = d['target_x'], d['target_y']
    m[y] = m[y] = m[y][:x] + 'B' + m[y][x + 1:]

    # Add obstacles (@,O)
    for k, block_val in d['blocks'].items():
        x, y = k.split('x')
        x, y = int(x), int(y)
        o_sign = '@' if block_val else 'O'
        m[y] = m[y] = m[y][:x] + o_sign + m[y][x + 1:]

    # Add switches (^)
    for k, _ in d['switches'].items():
        x, y = k.split('x')
        x, y = int(x), int(y)
        m[y] = m[y] = m[y][:x] + '^' + m[y][x + 1:]

    for mt in m:
        dprint(mt)
    dprint("")


def optimize_actions(s, max_pattern_length=8):
    def find_most_repeating_pattern(s: str, k: int) -> str:
        # Create a dictionary to store the count of each pattern
        # Keep track of the frequency of each pattern
        pattern_counts = defaultdict(int)

        # Initialize the sliding window
        window_start = 0
        window_end = k

        # Slide the window over the string, counting the frequency of each pattern
        while window_end <= len(s):
            pattern = s[window_start:window_end]
            pattern_counts[pattern] += 1
            window_start += 1
            window_end += 1

        # Find the pattern with the highest frequency
        most_common_pattern = max(pattern_counts, key=pattern_counts.get)
        repeats = pattern_counts[most_common_pattern]

        return most_common_pattern, repeats

    def find_best_string_to_reduce(s, max_pattern_length=8):
        hit = False
        best_pattern = None
        best_value = 0
        max_pattern_length = min(max_pattern_length, len(s))

        if len(s) < 4:
            return None, None

        for pattern_length in range(3, max_pattern_length):
            pattern, repeats = find_most_repeating_pattern(s, pattern_length)
            value = (repeats * pattern_length) - repeats
            if value > best_value:
                best_value = value
                best_pattern = pattern
                if hit:
                    pass
                    # break
                else:
                    hit = True
        return best_pattern, value

    sq = copy(s)
    funcs = dict()
    possible_functions = 9
    for i in range(1, possible_functions + 1):
        best_pattern, best_repeats = find_best_string_to_reduce(sq, max_pattern_length)
        if best_pattern is None:
            break
        funcs[i] = best_pattern
        sq = sq.replace(best_pattern, f'{i}')

    resulting_string = ';'.join([sq] + [funcs[i] for i in range(1, len(funcs) + 1)])
    resulting_value = len(resulting_string)

    if len(s) < resulting_value:
        return s, len(s)
    return resulting_string, resulting_value


class State:
    def __init__(self, d_start=None):
        global lines

        if d_start is None:
            # Read inputs from world
            self.turn = 0
            self.width, self.height = [int(i) for i in input().split()]
            lines = list()
            for i in range(self.height):
                lines += [input()]

            self.start_x, self.start_y = [int(i) for i in input().split()]
            self.target_x, self.target_y = [int(i) for i in input().split()]
            self.switch_count = int(input())
            self.terminal = False
            self.switches = dict()
            self.blocks = dict()
            self.balls_moves = 0
            for i in range(self.switch_count):
                # initial_state: 1 if blocking, 0 otherwise
                switch_x, switch_y, block_x, block_y, initial_state = [int(j) for j in input().split()]
                self.switches[f'{switch_x}x{switch_y}'] = f'{block_x}x{block_y}'
                self.blocks[f'{block_x}x{block_y}'] = bool(initial_state)

            self.actions = list()
            self.balls = dict()
            idx = 0
            for y, line in enumerate(lines):
                for x in range(len(line)):
                    if line[x] == '+':
                        self.balls[f'{x}x{y}'] = idx
                        lines[y] = lines[y][:x] + '.' + lines[y][x + 1:]
                        idx += 1
            self.switches_triggered = sum(self.blocks.values())

        else:
            # Read parameters from d_start
            self.turn = d_start['turn']
            self.actions = d_start['actions']
            self.width = d_start['width']
            self.height = d_start['height']
            # self.lines = d_start['lines']
            self.start_x = d_start['start_x']
            self.start_y = d_start['start_y']
            self.target_x = d_start['target_x']
            self.target_y = d_start['target_y']
            self.switch_count = d_start['switch_count']
            self.switches = d_start['switches']
            self.blocks = d_start['blocks']
            self.terminal = d_start['terminal']
            self.balls = d_start['balls']
            self.balls_moves = d_start['balls_moves']
            self.switches_triggered = d_start['switches_triggered']

        self.distance = abs(self.start_x - self.target_x) + abs(self.start_y - self.target_y)

    def export(self):
        # Convert all to dict
        d = dict()
        d['turn'] = self.turn
        d['width'] = self.width
        d['height'] = self.height
        d['start_x'] = self.start_x
        d['start_y'] = self.start_y
        d['target_x'] = self.target_x
        d['target_y'] = self.target_y
        d['switch_count'] = self.switch_count
        d['switches'] = self.switches.copy()
        d['blocks'] = self.blocks.copy()
        d['terminal'] = self.terminal
        d['balls'] = self.balls.copy()
        d['balls_moves'] = self.balls_moves
        d['switches_triggered'] = self.switches_triggered
        d['actions'] = self.actions.copy()
        return d

    def apply(self, action):

        type_of_copy = 'export'
        if type_of_copy == 'deepcopy':
            n_state = copy(self)
        elif type_of_copy == 'export':
            n_state = State(d_start=self.export())

        n_state.turn += 1
        n_state.actions.append(action[0])

        n_state.start_x, n_state.start_y = action[2], action[3]

        switch_sig = f'{n_state.start_x}x{n_state.start_y}'
        switch = n_state.switches.get(switch_sig, None)
        if switch is not None:
            block_state = n_state.blocks[switch]
            switch_old_state = block_state
            switch_new_state = not switch_old_state
            n_state.switches_triggered += int(switch_new_state) - int(switch_old_state)
            n_state.blocks[switch] = switch_new_state

        if action[1]:
            old_ball_sig = f'{n_state.start_x}x{n_state.start_y}'
            new_ball_sig = f'{action[4]}x{action[5]}'
            n_state.balls[new_ball_sig] = n_state.balls.pop(old_ball_sig)
            n_state.balls_moves += 1

            switch_sig = new_ball_sig
            switch = n_state.switches.get(switch_sig, None)
            if switch is not None:
                block_state = n_state.blocks[switch]
                switch_old_state = block_state
                switch_new_state = not switch_old_state
                n_state.switches_triggered += int(switch_new_state) - int(switch_old_state)
                n_state.blocks[switch] = switch_new_state

        if n_state.start_x == n_state.target_x and n_state.start_y == n_state.target_y:
            n_state.terminal = True

        n_state.distance = abs(n_state.start_x - n_state.target_x) + abs(n_state.start_y - n_state.target_y)
        return n_state

    def update(self, d=None):
        if d is None:
            # Get turn parameters from CLI
            pass
        else:
            # Get turn parameters from d
            pass
        return None

    def get_actions(self):
        ret = list()
        for action in ['U', 'D', 'L', 'R']:
            if action == 'U':
                x, y = self.start_x, self.start_y - 1
                next_x, next_y = self.start_x, self.start_y - 2
            elif action == 'D':
                x, y = self.start_x, self.start_y + 1
                next_x, next_y = self.start_x, self.start_y + 2
            elif action == 'L':
                x, y = self.start_x - 1, self.start_y
                next_x, next_y = self.start_x - 2, self.start_y
            elif action == 'R':
                x, y = self.start_x + 1, self.start_y
                next_x, next_y = self.start_x + 2, self.start_y

            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                continue
            next_mark = lines[y][x]
            if next_mark == '#':
                continue

            block = self.blocks.get(f'{x}x{y}', None)
            if block is not None:
                if block:
                    continue

            ball_sig = f'{x}x{y}'
            ball = self.balls.get(ball_sig, None)
            if ball is not None:
                continue
                next_next_mark = lines[next_y][next_x]
                if next_next_mark == '.':
                    # Check if Fry is not on the next next cell
                    if next_x == self.target_x and next_y == self.target_y:
                        continue

                    # Check if there is a ball in the next_next_mark
                    next_ball = self.balls.get(f'{next_x}x{next_y}', None)
                    if next_ball is None:
                        # The section one, "True" explain if this action pushes a block (True means push move)
                        ret.append((action, True, x, y, next_x, next_y))
                        continue
                continue

            # The section one, "False" explain if this action pushes a block (False means regular move)
            ret += [(action, False, x, y, next_x, next_y)]

        return ret

    def get_sig(self):
        sig = '@'.join([
            f'{self.start_x}x{self.start_y}',
            ''.join(self.balls.keys()),
            ''.join([str(int(v)) for v in self.blocks.values()]),
        ])

        return sig

    def get_rank(self):
        return (2 * self.switches_triggered) - (self.turn // 50) - (self.balls_moves // 4)

    def __gt__(self, other):
        return not self.__lt__(other)

    def __lt__(self, other):
        return self.get_rank() < other.get_rank()


class Agent:
    def __init__(self, state=None):
        self.terminals = dict()
        self.terminals_scores = dict()
        self.terminal_strings = dict()

    def recommend(self):
        if len(self.terminals) == 0:
            dprint('<<<<<< No terminals found >>>>>>')
            return 'MONKEY', 0, 'MONKEY'

        # Find the key of the minimal value of self.terminals_scores
        min_key = min(self.terminals_scores, key=self.terminals_scores.get)
        min_action = self.terminals[min_key]
        min_string = self.terminal_strings[min_key]
        return min_action, self.terminals_scores[min_key], min_string

    def observe(self, state=None):
        time_cap = 0.85
        turn_cap = 1000
        if LOCAL_MODE:
            time_cap = 50000  # 3.5
        curr_state = state
        q = PriorityQueue()
        q.put((0, curr_state))
        terminals = dict()
        terminal_scores = dict()
        terminal_strings = dict()
        visited = dict()
        round = -1
        max_turn = 0
        wins = 0
        allow_pushing = False
        break_flag = False
        while not q.empty() and not break_flag:
            round += 1
            # curr_state, curr_score = q.get()
            curr_score, curr_state = q.get()
            max_turn = max(max_turn, curr_state.turn)
            actions = curr_state.get_actions()

            # if LOCAL_MODE:
            #     stop_x, stop_y = 3, 3
            #     block_x, block_y = 7, 3
            #     block_state = curr_state.blocks.get(f'{block_x}x{block_y}', None)
            #     if curr_state.start_x == stop_x and curr_state.start_y == stop_y:
            #         print(f'breakpoint {curr_state.turn}')

            duration = time.time() - stime
            if duration > time_cap:
                break_flag = True
                dprint(f'Break (TIME) at round {round}')
                dprint(f"Solutions: {len(terminals)}")
                break
            if curr_state.turn > turn_cap:
                # dprint("State turn cap reached")
                continue

            if round % 2000 == 0:
                msg = ''
                msg += f'[Round {round}]'
                msg += f'[Queue {q.qsize()}]'
                msg += f'[Terminal {len(terminals)}]'
                msg += f'[Visited {len(visited)}]'
                msg += f'[Position: {curr_state.start_x}x{curr_state.start_y}]'
                # msg += f'[History: {len(curr_state.actions)}]'
                msg += f'[MAX Turn: {max_turn}]'
                msg += f'[Turn: {curr_state.turn}]'
                msg += f'[Balls moved: {curr_state.balls_moves}]'
                msg += f'[Switches triggered: {curr_state.switches_triggered}]'
                msg += f'[Distance: {curr_state.distance}]'
                msg += f'[Actions: {"-".join([a[0] for a in actions])}]'
                dprint(msg)

            for action in actions:
                next_state = curr_state.apply(action)
                sig = next_state.get_sig()

                if next_state.terminal:
                    terminal_s = ''.join(next_state.actions)
                    terminal_strings[sig] = terminal_s
                    ts, tv = optimize_actions(terminal_s)
                    if sig in terminals:
                        old_value = terminal_scores[sig]
                        if tv < old_value:
                            terminals[sig] = ts
                            terminal_scores[sig] = tv
                            terminal_strings[sig] = terminal_s
                    else:
                        terminals[sig] = ts
                        terminal_scores[sig] = tv
                        terminal_strings[sig] = terminal_s
                    # dprint(f"Terminal state found: {sig}\tValue: {tv}\t{terminal_s}")
                    if LOCAL_MODE:
                        msg = ''
                        msg += f'<><><><><><>'
                        msg += f'[Round {round}]'
                        msg += f'[Queue {q.qsize()}]'
                        msg += f'[Terminal {len(terminals)}]'
                        msg += f'[Visited {len(visited)}]'
                        msg += f'[Position: {curr_state.start_x}x{curr_state.start_y}]'
                        # msg += f'[History: {len(curr_state.actions)}]'
                        msg += f'[MAX Turn: {max_turn}]'
                        msg += f'[Turn: {curr_state.turn}]'
                        msg += f'[Balls moved: {curr_state.balls_moves}]'
                        msg += f'[Switches triggered: {curr_state.switches_triggered}]'
                        msg += f'[Distance: {curr_state.distance}]'
                        msg += f'[Actions: {"-".join([a[0] for a in actions])}]'
                        dprint(msg)
                    else:
                        dprint(f'<><> WINS: {wins}')
                    break_flag = True
                    continue
                    # break
                else:
                    if sig in visited:
                        continue
                    wins += 1
                    visited[sig] = next_state
                    
                    q.put((next_state.get_rank(), next_state))

        self.terminal_strings = terminal_strings
        self.terminals = terminals
        self.terminals_scores = terminal_scores
        return self.recommend()


d = None
if LOCAL_MODE:
    lines = ['#####################', '#.#.#.#.........#.#.#', '#.#.#.#.###.###.#.#.#', '#.......#...#...#...#',
             '#####.#.#.#.#.#.###.#', '#.#.#.#...#.........#', '#.#.#.#.#.##..#####.#', '#.........#.#...#...#',
             '########..#.###.###.#', '#.....#.#.....#.#...#', '#.###.#.###.#.#.###.#', '#.#.#.....#.......#.#',
             '#.#.####..#######...#', '#...#...#.#.#.#...#.#', '###.###.#.#.#.....#.#', '#...#...#...#.#.#...#',
             '#.#.###.#...#...###.#', '#.#.#...#.#.#.#.#.#.#', '#.#.###.#.#.#.#.#.#.#', '#...................#',
             '#####################']
    d = dict()
    d["turn"] = 0
    d["width"] = 21
    d["height"] = 21
    d["start_x"] = 15
    d["start_y"] = 17
    d["target_x"] = 1
    d["target_y"] = 18
    d["switch_count"] = 11
    d["switches"] = {'15x5': '8x19', '11x16': '6x11', '9x11': '7x14', '13x15': '7x1', '5x13': '5x7', '17x1': '3x18',
                     '15x9': '2x3', '4x19': '13x18', '19x18': '9x3', '3x7': '3x19', '15x11': '15x4'}
    d["blocks"] = {'8x19': True, '6x11': True, '7x14': True, '7x1': True, '5x7': True, '3x18': True, '2x3': True,
                   '13x18': False, '9x3': False, '3x19': True, '15x4': False}
    d["terminal"] = False
    d["balls"] = {'11x2': 0, '9x4': 1, '13x4': 2, '8x12': 3, '16x13': 4, '14x14': 5, '10x15': 6, '1x16': 7, '10x16': 8,
                  '14x16': 9}
    d["balls_moves"] = 0
    d["switches_triggered"] = 8
    d["actions"] = []

state = State(d)
dprint("lines = " + str(lines))
print_dict(state.export())
agent = Agent()
stime = time.time()

if True:
    if LOCAL_MODE:
        d_start = dict()
        pass
    else:
        state.update(d)

    print_dict(state.export())
    dprint("")

    action, action_value, action_str = agent.observe(state)

    dprint("Run time: {:.2f}s".format(time.time() - stime))
    dprint("Expected value: " + str(action_value))
    dprint("Full Action: " + action_str)
    dprint("Full Action length: " + str(len(action_str)))
    dprint("Action: " + action)

    if LOCAL_MODE:
        time.sleep(0.1)
    print(action)
