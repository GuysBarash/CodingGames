import sys
import math
from datetime import datetime
import numpy as np
import pickle as p
from copy import copy
from collections import deque
import socket
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


def char_to_int(c):
    if c == ' ':
        return 0
    else:
        return ord(c) - 64


def int_to_char(d):
    d = d % 27
    if d == 0:
        return ' '
    else:
        return chr(64 + d)


class SimpleBilbo:
    def __init__(self):
        self.char_max_value = 27

    def act(self, forest):
        s_action = ''
        if forest.step == 0:
            nforest = forest
        else:
            s_action += '>'
            nforest = forest.apply('>')

        curr = nforest.current_phrase
        trgt = nforest.magic_phrase
        trgt_chr = trgt[len(curr)]
        trgt_int = char_to_int(trgt_chr)

        current_int = nforest.forest[nforest.bilbo_position]
        current_char = int_to_char(current_int)
        s_action += self._optimal_step(current_int, trgt_int)
        s_action += '.'

        return s_action

    def _optimal_step(self, src_int, trgt_int):
        upped_src = src_int + self.char_max_value
        down_steps = (upped_src - trgt_int) % 27
        up_steps = (trgt_int - src_int) % 27
        if up_steps > down_steps:
            return down_steps * '-'
        else:
            return up_steps * '+'


class GreedyBilbo:
    def __init__(self):
        self.char_max_value = 27
        self.forest_size = 30

    def act(self, forest):

        curr = forest.current_phrase
        trgt = forest.magic_phrase
        trgt_chr = trgt[len(curr)]
        trgt_int = char_to_int(trgt_chr)

        current_pos = forest.bilbo_position

        # Optimal shift in cell
        current_forest = forest.forest
        current_forest_t = [self._optimal_step(src_int, trgt_int) for src_int in current_forest]
        current_forest_delta = [t[0] for t in current_forest_t]

        # Optimal moves to shift cell
        moves_to_position = [self._optimal_direction(current_pos, i) for i in range(forest.forest_size)]
        current_forest_moves = [current_forest_delta[i] + moves_to_position[i][0] for i in
                                range(forest.forest_size)]

        # find optimal
        optimal_pos = np.argmin(current_forest_moves)
        optimal_movement = moves_to_position[optimal_pos][1] * moves_to_position[optimal_pos][0]

        optimal_shift = current_forest_t[optimal_pos][1] * current_forest_t[optimal_pos][0]

        s_action = ''
        s_action += optimal_movement
        s_action += optimal_shift
        s_action += '.'

        return s_action

    def _optimal_step(self, src_int, trgt_int):
        upped_src = src_int + self.char_max_value
        down_steps = (upped_src - trgt_int) % self.char_max_value
        up_steps = (trgt_int - src_int) % self.char_max_value
        if up_steps > down_steps:
            return down_steps, '-'
        else:
            return up_steps, '+'

    def _optimal_direction(self, src_pos, trt_pos):
        upped_src = src_pos + self.forest_size
        down_steps = (upped_src - trt_pos) % self.forest_size
        up_steps = (trt_pos - src_pos) % self.forest_size
        if up_steps > down_steps:
            return down_steps, '<'
        else:
            return up_steps, '>'


class Forest:

    def __init__(self, d=None, magic_phrase=None):
        if d is None:
            self.step = 0
            self.bilbo_position = 0
            self.forest_size = 30
            self.initial_value = 0
            self.char_max_value = 27
            self.forest = [self.initial_value] * self.forest_size
            self.current_phrase = ''
            self.magic_phrase = magic_phrase

        else:
            self.step = d['step']
            self.bilbo_position = d['bilbo_position']
            self.forest_size = d['forest_size']
            self.initial_value = d['initial_value']
            self.forest = d['forest']
            self.char_max_value = d['char_max_value']
            self.magic_phrase = d['magic_phrase']
            self.current_phrase = d['current_phrase']

        self.terminal = False
        self.winner = None
        self.forest_chr = None
        self._translate_forest()

    def get_params(self):
        d = dict()
        d['step'] = self.step
        d['bilbo_position'] = self.bilbo_position
        d['forest_chr'] = self.forest_chr
        d['forest'] = self.forest
        d['forest_size'] = self.forest_size
        d['char_max_value'] = self.char_max_value
        d['initial_value'] = self.initial_value
        d['current_phrase'] = self.current_phrase
        d['magic_phrase'] = self.magic_phrase
        d['terminal'] = self.terminal
        d['winner'] = self.winner

        return d

    def _translate_forest(self):
        self.forest_chr = ''.join([int_to_char(d) for d in self.forest])

    def _check_terminal(self):
        if len(self.current_phrase) > len(self.magic_phrase):
            self.terminal = True
            self.winner = False
        elif len(self.current_phrase) == len(self.magic_phrase):
            self.terminal = True
            self.winner = self.current_phrase == self.magic_phrase
        else:
            sub_magic_phrase = self.magic_phrase[:len(self.current_phrase)]
            if sub_magic_phrase == self.current_phrase:
                self.terminal = False
                self.winner = None
            else:
                self.terminal = True
                self.winner = False

    def update(self, action):
        pos = self.bilbo_position
        if action in ['+', '-']:
            d = {'+': +1, '-': -1}
            self.forest[pos] = (self.forest[pos] + d[action]) % self.char_max_value
        elif action in ['<', '>']:
            d = {'>': +1, '<': -1}
            self.bilbo_position = (pos + d[action]) % self.forest_size
        elif action == '.':
            c_int = self.forest[pos]
            c_char = int_to_char(c_int)
            self.current_phrase += c_char
            self._check_terminal()

        else:
            raise Exception(f"BAD ACTION! <{action}>")

    def apply(self, action):
        forest = Forest(self.get_params())
        for t_action in action:
            forest.update(t_action)

        forest.step += 1
        return forest

    def display(self, chars=True):
        arr = self.forest
        if chars:
            arr = [int_to_char(d) for d in arr]

        arrs = [f' {t} ' for t in arr]
        arrs[self.bilbo_position] = f'<{arr[self.bilbo_position]}>'
        s = '|'.join(arrs)
        s = f'[{self.step:>3}] ' + s + f'| [{self.current_phrase}]'
        dprint(s)


if LOCAL_MODE:
    d = dict()
    magic_phrase = "ABCDEFGHIJKLMNOPQRSTUVWXYZAABCDEFGHIJKLMNOPQRSTUVWXYZAABCDEFGHIJKLMNOPQRSTUVWXYZAABCDEFGHIJKLMNOPQRSTUVWXYZAABCDEFGHIJKLMNOPQRSTUVWXYZAABCDEFGHIJKLMNOPQRSTUVWXYZAABCDEFGHIJKLMNOPQRSTUVWXYZAABCDEFGHIJKLMNOPQRSTUVWXYZAABCDEFGHIJKLMNOPQRSTUVWXYZAABCDEFGHIJKLMNOPQRSTUVWXYZAABCDEFGHIJKLMNOPQRSTUVWXYZA"

else:
    magic_phrase = input()

magic_values = [char_to_int(c) for c in magic_phrase]
dprint(f"Sentence:\n\"{magic_phrase}\"")
dprint(f"Values: {magic_values}")

forest = Forest(magic_phrase=magic_phrase)
bilbo = GreedyBilbo()
print_dict(forest.get_params())

forest.display()
s = ''
while not forest.terminal:
    action = bilbo.act(forest)
    s += action
    forest = forest.apply(action)
    if DEBUG_MODE:
        forest.display()

print_dict(forest.get_params())

dprint(f"Instructions: (Length: {len(s)})")
if LOCAL_MODE:
    time.sleep(0.1)
print(s)
