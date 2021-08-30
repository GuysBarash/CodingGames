import sys
import math
import socket

import numpy as np
import pandas as pd

import pickle as p
from copy import deepcopy
from collections import deque
from queue import PriorityQueue
from datetime import datetime
import time

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
        self.n = None
        self.raw = None
        self.arr = None
        self.sol = None

    def update(self, d=None):
        read_time = datetime.now()
        if d is None:
            self.n = int(input())
            self.raw = [input() for _ in range(self.n)]
        else:
            self.n = d['n']
            self.raw = d['raw']

        self.arr = np.array([list(t.ljust(55)) for t in self.raw + ['', '', '']])
        read_time = (datetime.now() - read_time).total_seconds()
        dprint(f"Read time: {read_time}")

    def solve(self):
        self.sol = self.arr.copy()
        for y in range(self.arr.shape[0]):
            for x in range(self.arr.shape[1] - 2):
                if self.arr[y, x] != ' ':
                    if self.sol[y + 1, x + 1] == ' ':
                        self.sol[y + 1, x + 1] = '-'
                        if self.sol[y + 2, x + 2] == ' ':
                            self.sol[y + 2, x + 2] = '`'

    def print_solution(self):
        for y in range(self.sol.shape[0] - 1):
            r = self.sol[y, :]
            broken = False
            for x in range(len(r)):
                if all(r[x:] == ' '):
                    pr = r[:x]
                    pr = ''.join(pr)
                    print(pr)
                    broken = True
                    break
                    # if pr == '':
                    #     pass
                    # else:
                    #     print(pr)
                    #     break
            if not broken:
                pr = ''.join(r)
                print(pr)

    def get_params(self):
        d = dict()
        d['n'] = self.n
        d['raw'] = self.raw
        time.sleep(0.2)
        return d


d = None
if LOCAL_MODE:
    d = dict()
    d["n"] = 15
    d["raw"] = [' ######   #######  ########  #### ##    ##  ######',
                '##    ## ##     ## ##     ##  ##  ###   ## ##    ##', '##       ##     ## ##     ##  ##  ####  ## ##',
                '##       ##     ## ##     ##  ##  ## ## ## ##   ####',
                '##       ##     ## ##     ##  ##  ##  #### ##    ##',
                '##    ## ##     ## ##     ##  ##  ##   ### ##    ##',
                ' ######   #######  ########  #### ##    ##  ######', '',
                ' ######      ###    ##     ## ########  ######', '##    ##    ## ##   ###   ### ##       ##    ##',
                '##         ##   ##  #### #### ##       ##', '##   #### ##     ## ## ### ## ######    ######',
                '##    ##  ######### ##     ## ##             ##', '##    ##  ##     ## ##     ## ##       ##    ##',
                ' ######   ##     ## ##     ## ########  ######']

world = World()
world.update(d)
print_dict(world.get_params())
world.solve()

if LOCAL_MODE:
    dprint_array(world.arr)
    dprint_array(world.sol)
time.sleep(0.2)
world.print_solution()
