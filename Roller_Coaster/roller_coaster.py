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
        self.memory = dict()
        self.memory_used = False
        self.capacity = None
        self.loops = None
        self.groups_count = None
        self.groups = list()
        self.money = 0
        self.turn = 0
        self.cycle = -1
        self.q = None

    def update(self, d=None):
        read_time = datetime.now()
        if d is None:
            self.capacity, self.loops, self.groups_count = [int(i) for i in input().split()]
            self.groups = [int(input()) for i in range(self.groups_count)]
            self.money = 0
            self.turn = 0
        else:
            self.capacity = d['capacity']
            self.loops = d['loops']
            self.groups_count = d['groups_count']
            self.groups = d['groups']
            self.money = d['money']
            self.turn = d['turn']

        read_time = (datetime.now() - read_time).total_seconds()
        dprint(f"Read time: {read_time}")

    def get_params(self):
        d = dict()
        d['capacity'] = self.capacity
        d['loops'] = self.loops
        d['groups_count'] = self.groups_count
        d['groups'] = self.groups
        d['money'] = self.money
        return d

    def loop(self):
        self.loops -= 1
        self.turn += 1
        q = deque(self.groups)

        nq = deque([])
        cap = 0
        while True:
            if len(q) <= 0 or (cap + q[0] > self.capacity):
                break
            else:
                a = q.popleft()
                cap += a
                nq.append(a)
        self.groups = list(q + nq)
        self.money += cap
        # dprint(f"[Loop {self.loops:>3}]\tMoney: {self.money}")

    def done(self):
        return self.loops == 0

    def remember(self):
        if self.memory_used:
            return None

        sig = '_'.join([str(t) for t in self.groups])
        if sig in self.memory:
            dprint(f"Loop!!!! {self.turn}")
            loop_value_t, loop_size_t = self.memory[sig]
            loop_value = self.money - loop_value_t
            loop_size = self.turn - loop_size_t

            loops_remain = self.loops // loop_size
            money_to_earn = loops_remain * loop_value

            self.turn += loops_remain
            self.loops -= (loops_remain * loop_size)
            self.money += money_to_earn
            self.memory_used = True

        else:
            self.memory[sig] = self.money, self.turn


d = None
if LOCAL_MODE:
    d = dict()
    d["capacity"] = 5
    d["loops"] = 350
    d["groups_count"] = 4
    d["groups"] = [2, 3, 5, 3]
    d['money'] = 0
    d['turn'] = 0

world = World()
world.update(d)
print_dict(world.get_params())
while True:
    world.remember()
    if world.done():
        break
    world.loop()
    print_dict(world.get_params())

print(world.money)
