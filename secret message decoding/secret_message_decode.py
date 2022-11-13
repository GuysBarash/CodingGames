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
LOCAL_MODE = socket.gethostname() == 'Barash-pc'


def string_to_ascii(s):
    return [ord(c) for c in s]


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


class World:
    def __init__(self, d=None):
        self.lines_count = None
        self.lines = None

    def update(self, d=None):
        if d is None:
            self.hs = int(input())
            self.ms = int(input())
            self.rows = list()
            for i in range(self.hs):
                row = input()
                self.rows.append(row)
        else:
            self.hs = d['hs']
            self.ms = d['ms']
            self.rows = d['rows']

    def get_params(self):
        d = dict()
        d['hs'] = self.hs
        d['ms'] = self.ms
        d['rows'] = self.rows
        return d

    def run(self):
        rows = self.rows
        rows = [string_to_ascii(row) for row in rows]
        r = np.array(rows)

        header = r[:self.hs, :self.hs]
        message = r[:,self.hs:]
        return None


d = None
if LOCAL_MODE:
    d = dict()
    d["hs"] = 2
    d["ms"] = 1
    d["rows"] = ['cg$', 'l?(']

world = World()
world.update(d)
print_dict(world.get_params())
res = world.run()
print(res)
