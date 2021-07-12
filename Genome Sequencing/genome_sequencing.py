import sys
import math
import socket
import difflib
import numpy as np
import pandas as pd
import itertools

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


# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.


def char_to_int(c):
    d = dict()
    d['G'] = 1
    d['A'] = 2
    d['T'] = 3
    d['C'] = 4
    return d[c]


class World:
    def __init__(self):
        self.items = 0
        self.seqs = list()

    def update(self, d=None):
        if d is None:
            self.items = int(input())
            for i in range(self.items):
                self.seqs.append(input())
        else:
            self.items = d['items']
            self.seqs = d['seqs']

    def get_params(self):
        d = dict()
        d['items'] = self.items
        d['seqs'] = self.seqs
        return d

    def swallow_substrings(self):
        words_to_remove = set()
        for i in range(len(self.seqs)):
            s1 = self.seqs[i]
            for j in range(len(self.seqs)):
                if i == j:
                    continue
                else:
                    s2 = self.seqs[j]
                    if s1 in s2:
                        if s1 == s2:
                            if j in words_to_remove:
                                pass
                            else:
                                words_to_remove.add(i)
                        else:
                            words_to_remove.add(i)

        for i in list(words_to_remove):
            rem = self.seqs[i]
            self.seqs[i] = '@'
            dprint(f"Word {rem} is a substring of another word. omitted.")
        self.seqs = [t for t in self.seqs if t != '@']

    def run(self):
        self.swallow_substrings()

        possible_permutations = get_permutations(self.seqs)
        optimal_key = dict()
        permutations_count = len(possible_permutations)
        for idx, seqs in enumerate(possible_permutations):
            s = ''
            for st in seqs:
                s = max_joint(s, st)
            optimal_key[idx] = s

        winner_idx, winner = min(optimal_key.items(), key=lambda q: len(q[1]))
        winner_l = len(winner)
        for i in range(permutations_count):
            optimal = optimal_key[i]
            msg = f'[{i+1}/{permutations_count}] {" --> ".join(possible_permutations[i])} {optimal} = {len(optimal)}'
            if i == winner_idx:
                msg += ' <----------'
            dprint(msg)

        dprint("")

        return winner_l


def max_joint(left_s, right_s):
    max_len = 0
    pointer = 0
    for i in range(1, len(right_s)):
        r_part = right_s[:i]
        l_part = left_s[-i:]
        if r_part == l_part:
            if len(r_part) > max_len:
                pointer = i
                max_len = len(r_part)

    ar_pointer = pointer
    ar_len = max_len
    ar_s = left_s + right_s[pointer:]
    return ar_s


def get_permutations(l):
    l_perms = list(itertools.permutations(l))
    return l_perms


code_section = True
if code_section:
    d = None
    if LOCAL_MODE:
        d = dict()
        d["items"] = 2
        d["seqs"] = ['AC', 'AC']

    world = World()
    world.update(d)
    print_dict(world.get_params())

    result = world.run()

print(result)
