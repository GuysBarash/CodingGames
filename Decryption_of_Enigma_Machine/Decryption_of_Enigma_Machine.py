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
        self.operation = None
        self.rotors = None
        self.pseudo_random_number = None
        self.sol = None

    def update(self, d=None):
        read_time = datetime.now()
        if d is None:
            self.operation = input()
            self.pseudo_random_number = int(input())
            self.rotors = [input() for _ in range(3)]
            self.message = input()
        else:
            self.operation = d['operation']
            self.pseudo_random_number = d['pseudo_random_number']
            self.rotors = d['rotors']
            self.message = d['message']

        read_time = (datetime.now() - read_time).total_seconds()
        dprint(f"Read time: {read_time}")

    def solve(self):
        rotor_0 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        rotor_0_9z = {idx: c for idx, c in enumerate(rotor_0)}
        rotor_0_z9 = {c: idx for idx, c in enumerate(rotor_0)}
        self.rotors = [rotor_0] + self.rotors

        if self.operation == 'ENCODE':
            s0 = self.message
            s1 = ''
            for i in range(len(s0)):
                c0 = s0[i]
                c0num = rotor_0_z9[c0]
                c1num = c0num + i + self.pseudo_random_number
                c1 = rotor_0_9z[c1num % len(rotor_0_9z)]
                s1 += c1

            s = s1
            for rotor_idx in range(3):
                st = ''
                rotor_0_9z = {idx: c for idx, c in enumerate(self.rotors[0])}
                rotor_0_z9 = {c: idx for idx, c in enumerate(self.rotors[0])}

                rotor_1_9z = {idx: c for idx, c in enumerate(self.rotors[rotor_idx + 1])}
                rotor_1_z9 = {c: idx for idx, c in enumerate(self.rotors[rotor_idx + 1])}

                for i in range(len(s)):
                    c0 = s[i]
                    c0num = rotor_0_z9[c0]
                    c1num = c0num
                    c1 = rotor_1_9z[c1num % len(rotor_1_9z)]
                    st += c1
                s = st
            self.sol = s


        else:
            s = self.message
            for rotor_idx in range(3, 0, -1):
                st = ''
                rotor_1_9z = {idx: c for idx, c in enumerate(self.rotors[0])}
                rotor_1_z9 = {c: idx for idx, c in enumerate(self.rotors[0])}

                rotor_0_9z = {idx: c for idx, c in enumerate(self.rotors[rotor_idx])}
                rotor_0_z9 = {c: idx for idx, c in enumerate(self.rotors[rotor_idx])}

                for i in range(len(s)):
                    c0 = s[i]
                    c0num = rotor_0_z9[c0]
                    c1num = c0num
                    c1 = rotor_1_9z[c1num % len(rotor_1_9z)]
                    st += c1
                s = st

            rotor = self.rotors[0]
            rotor_9z = {idx: c for idx, c in enumerate(rotor)}
            rotor_z9 = {c: idx for idx, c in enumerate(rotor)}
            st = ''
            for i in range(len(s)):
                c0 = s[i]
                c0num = rotor_z9[c0]
                c1num = (c0num - self.pseudo_random_number - i)
                c1 = rotor_9z[c1num % len(rotor_9z)]
                st += c1
            s = st
            self.sol = s

    def print_solution(self):
        print(self.sol)

    def get_params(self):
        d = dict()
        d['operation'] = self.operation
        d['pseudo_random_number'] = self.pseudo_random_number
        d['rotors'] = self.rotors
        d['message'] = self.message
        time.sleep(0.2)
        return d


d = None
if LOCAL_MODE:
    d = dict()
    d["operation"] = "DECODE"
    d["pseudo_random_number"] = 4
    d["rotors"] = ['BDFHJLCPRTXVZNYEIWGAKMUSQO', 'AJDKSIRUXBLHWTMCQGZNPYFVOE', 'EKMFLGDQVZNTOWYHXUSPAIBRCJ']
    d["message"] = "KQF"

world = World()
world.update(d)
print_dict(world.get_params())
world.solve()

time.sleep(0.1)
world.print_solution()
