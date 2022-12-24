import sys
import math
import socket
import re

import numpy as np
import pandas as pd

import pickle as p
from copy import copy
from collections import deque
from datetime import datetime
import time
from collections import deque
from queue import PriorityQueue

DEBUG_MODE = True
LOCAL_MODE = socket.gethostname() == 'Barash-pc'


def dprint(s=''):
    print(s, file=sys.stderr, flush=True)


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
    def __init__(self, d_start=None):
        if d_start is None:
            # Read inputs from world
            self.s = input()

        else:
            # Read parameters from d_start
            self.s = d_start['s']

    def export(self):
        # Convert all to dict
        d = dict()
        d['s'] = self.s
        return d

    def update(self, d=None):
        if d is None:
            # Get turn parameters from CLI
            pass
        else:
            # Get turn parameters from d
            pass
        return None


class Agent:
    def __init__(self, state=None):
        pass

    def observe(self, state=None):
        s = state.s

        action = None
        return action


d = None
if LOCAL_MODE:
    d = dict()
    d["s"] = "Frankie"

state = State(d)
print_dict(state.export())
agent = Agent()

if True:
    if LOCAL_MODE:
        d_start = dict()
        pass
    else:
        state.update(d)

    print_dict(state.export())
    dprint("")

    action = agent.observe(state)
    print(action)
