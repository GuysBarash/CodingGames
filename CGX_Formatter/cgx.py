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


class World:
    def __init__(self, d=None):
        self.lines_count = None
        self.lines = None

    def update(self, d=None):
        if d is None:
            self.lines_count = int(input())
            self.lines = list()
            for i in range(self.lines_count):
                cgxline = input()
                self.lines.append(cgxline)
        else:
            self.lines_count = d['lines_count']
            self.lines = d['lines']

        self.raw = ''.join(self.lines)

    def get_params(self):
        d = dict()
        d['lines_count'] = self.lines_count
        d['lines'] = self.lines
        return d

    def run(self):

        return self.raw


d = None
if LOCAL_MODE:
    d = dict()
    d["lines_count"] = 4
    d["lines"] = ['  ', '', '\t true', '']

world = World()
world.update(d)
print_dict(world.get_params())
res = world.run()
print(res)
