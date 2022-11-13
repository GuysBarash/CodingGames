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
        self.adjList = "Adaptable Adventurous Affectionate Courageous Creative Dependable Determined Diplomatic Giving Gregarious Hardworking Helpful Hilarious Honest Non-judgmental Observant Passionate Sensible Sensitive Sincere"
        self.adjList = self.adjList.lower().split(' ')
        self.goodList = "Love, Forgiveness, Friendship, Inspiration, Epic Transformations, Wins"
        self.goodList = self.goodList.lower().replace(', ', ',').split(',')
        self.badList = "Crime, Disappointment, Disasters, Illness, Injury, Investment Loss"
        self.badList = self.badList.lower().replace(', ', ',').split(',')

        self.vowels = list('aeiouy')
        self.consonants = list('bcdfghjklmnpqrstvwxz')

    def observe(self, state=None):
        s = state.s
        name = state.s
        s = s.lower()
        s = ''.join(filter(str.isalpha, s))

        vowels = [c for c in s if c in self.vowels]
        # vowels = list(dict.fromkeys(vowels))
        vowels_idx = [self.vowels.index(v) for v in vowels]
        conso = [c for c in s if c in self.consonants]
        conso = list(dict.fromkeys(conso))
        conso_idx = [self.consonants.index(c) for c in conso]

        err = True
        if len(conso_idx) >= 3 and len(vowels_idx) >= 2:
            if len(self.adjList) >= max(conso_idx[:3]):
                if len(self.goodList) >= vowels_idx[0]:
                    if len(self.badList) >= vowels_idx[1]:
                        err = False

        if err:
            return f'Hello {name}.'
        else:
            w1 = self.adjList[conso_idx[0]]
            w2 = self.adjList[conso_idx[1]]
            w3 = self.adjList[conso_idx[2]]
            w4 = self.goodList[vowels_idx[0]]
            w5 = self.badList[vowels_idx[1]]

            l1 = f"It's so nice to meet you, my dear {w1} {name}."
            l2 = f"I sense you are both {w2} and {w3}."
            l3 = f"May our future together have much more {w4} than {w5}."
            action = '\n'.join([l1, l2, l3])
            return action


d = None
if LOCAL_MODE:
    d = dict()
    d["s"] = "Meg Eagleton"

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
