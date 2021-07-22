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
    def __init__(self):
        self.rooms_count = None
        self.rooms = None

        self.maparr = None
        self.mapdf = None

    def update(self, d=None):
        read_time = datetime.now()
        if d is None:
            self.rooms_count = int(input())
            self.rooms = [list()] * self.rooms_count
            for i in range(self.rooms_count):
                self.rooms[i] = input().split()
            dprint("Update completed.")
        else:
            self.rooms_count = d['rooms_count']
            self.rooms = d['rooms']

        self._parse_input()
        read_time = (datetime.now() - read_time).total_seconds()
        dprint(f"Read time: {read_time}")

    def get_params(self):
        d = dict()
        d['rooms_count'] = self.rooms_count
        d['rooms'] = self.rooms
        return d

    def _parse_input(self):
        self.mapdf = pd.DataFrame(index=range(self.rooms_count), columns=['idx', 'money', 'room1', 'room2'],
                                  data=self.rooms)
        self.mapdf['money'] = self.mapdf['money'].astype(int)
        for room in ['room1', 'room2']:
            self.mapdf.loc[self.mapdf[room].eq('E'), room] = self.rooms_count
            self.mapdf[room] = self.mapdf[room].astype(int)
        self.mapdf = self.mapdf.astype(int)
        self.mapdf.loc[self.rooms_count] = [self.rooms_count, 0, -9, -9]
        self.maparr = self.mapdf.to_numpy()


class Topological_agent:
    def __init__(self):
        self.mapdf = None
        self.nodes = None
        self.end_ptr = None

    def observe(self, world):
        self.maparr = world.maparr
        self.end_ptr = world.rooms_count
        self.run()
        return self.nodes[self.end_ptr]

    def run(self):
        nodes = dict()
        nodes[0] = self.maparr[0, 1]

        dprint("Running search")
        stime = datetime.now()
        total_count = self.maparr.shape[0]
        for idx, (ptr, money_in_ptr, n1, n2) in enumerate(self.maparr):
            if ptr % 500 == 0:
                dprint(f"ptr = {idx+1}/{total_count}\t Time: {(datetime.now() - stime)}")
            if ptr == self.end_ptr:
                continue
            current_worth = nodes.get(ptr, 0)
            nsworth = current_worth + self.maparr[[n1, n2], 1]

            for idx, n in enumerate([n1, n2]):
                new_worth = nsworth[idx]
                nodes[n] = np.max([nodes.get(n, 0), new_worth])

        self.nodes = nodes


d = None
if LOCAL_MODE:
    d = dict()
    d["rooms_count"] = 1
    d["rooms"] = [['0', '200', 'E', 'E']]

world = World()
world.update(d)
# print_dict(world.get_params())

agent = Topological_agent()
res = agent.observe(world)
print(res)
