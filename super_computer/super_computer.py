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
pd.options.mode.chained_assignment = None


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
    def __init__(self):
        self.count = None
        self.table = None

        self.df = None

    def update(self, d=None):
        if d is None:
            self.count = int(input())
            self.table = list()
            self.df = pd.DataFrame(index=range(self.count * 2), columns=['ID', 'START', 'DURATION', 'TOKEN', 'OVERLAP'],
                                   data=0)
            for i in range(self.count):
                j, d = [int(j) for j in input().split()]
                self.table.append((j, d))
        else:
            self.count = d['count']
            self.table = d['table']

    def get_params(self):
        d = dict()
        d['count'] = self.count
        d['table'] = self.table
        return d

    def pop_idx(self, df, idx):
        df = df[~df['ID'].eq(idx)]
        df['OVERLAP'] = df['TOKEN'].cumsum()
        return df

    def run(self):
        df = pd.DataFrame(self.table, columns=['START', 'DURATION'])
        df['ID'] = df.index
        df['START'] *= 2
        df['DURATION'] *= 2
        df['END'] = df['START'] + df['DURATION'] - 1
        df['T'] = df['START']
        df = df.sort_values(by=['END'])
        df['prev_end'] = df['END'].shift(+1).fillna(-1).astype(int)

        r = 0
        while True:
            r += 1
            df['prev_end'] = df['END'].shift(+1).fillna(-1).astype(int)
            df['LEGAL'] = df['START'] > df['prev_end']
            df['PREV_LEGAL'] = df['LEGAL'].shift(+1).fillna(True).astype(bool)
            df['REMOVE'] = (~df['LEGAL']) & df['PREV_LEGAL']
            remove_count = df['REMOVE'].sum()
            if remove_count > 0:
                df = df[~df['REMOVE']]
                dprint(f'[{r}]\tRemoving: {remove_count}')
            else:
                break
        return df.shape[0]


world = World()


def random_input(count=99, days_window=300, max_duration=20, seed=None):
    if seed is not None:
        np.random.seed(seed)

    d = dict()
    d['count'] = count
    s = np.random.randint(1, days_window, size=count)
    du = np.random.randint(1, max_duration, size=count)
    d['table'] = [(s[i], du[i]) for i in range(count)]
    return d


d = None
if LOCAL_MODE:
    d = dict()
    d["count"] = 5
    d["table"] = [(3, 5), (9, 2), (24, 5), (16, 9), (11, 6)]
    d = random_input(count=9999, days_window=2000, max_duration=30, seed=10)

world.update(d)
# print_dict(world.get_params())
res = world.run()
print(res)
