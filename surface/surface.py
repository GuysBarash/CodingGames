import sys
import math
import socket

import numpy as np
import pandas as pd

import pickle as p
from copy import copy
from queue import Queue
from datetime import datetime
import time
import itertools

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


def cor_2_str(x, y):
    return f'{x}_{y}'


def str_2_cor(s):
    return [int(t) for t in s.split('_')]


class World:
    def __init__(self):
        self.width = None
        self.height = None
        self.map_raw = None
        self.map = None

        self.coordinates_count = None
        self.coordinates = None

        self.lakes = None
        self.lakes_count = 0

        self.results = list()

    def update(self, d=None):
        if d is None:
            self.width = int(input())
            self.height = int(input())
            self.map_raw = list()
            for i in range(self.height):
                self.map_raw.append(input())

            self.coordinates_count = int(input())
            self.coordinates = list()
            for i in range(self.coordinates_count):
                x, y = [int(j) for j in input().split()]
                self.coordinates.append((x, y))
        else:
            self.width = d['width']
            self.height = d['height']
            self.map_raw = d['map_raw']

            self.coordinates_count = d['coordinates_count']
            self.coordinates = d['coordinates']

    def get_params(self):
        d = dict()
        d['width'] = self.width
        d['height'] = self.height
        d['map_raw'] = self.map_raw
        d['coordinates_count'] = self.coordinates_count
        d['coordinates'] = self.coordinates
        return d

    def pre_calc(self):
        l1 = range(self.width)
        l2 = range(self.height)
        l = [cor_2_str(*t) for t in list(itertools.product(l1, l2))]
        self.lakes = pd.DataFrame(index=l, columns=[-1], data=0)
        self.lakes_count = 0
        self.map = pd.DataFrame(columns=l1, index=l2,
                                data=[list(t) for t in self.map_raw]
                                )

    def valid_coors(self, x, y):
        if 0 <= x < self.width:
            if 0 <= y < self.height:
                return True
        return False

    def run_coordinate(self, x, y):
        current_lake = self.lakes_count
        self.lakes_count += 1
        self.lakes[current_lake] = 0
        q = Queue()
        visited = dict()
        q.put((x, y, cor_2_str(x, y)))
        while not q.empty():
            tx, ty, t_sig = q.get()
            self.lakes.loc[t_sig, current_lake] = 1

            # Up
            if ty > 0:
                ny = ty - 1
                nx = tx
                nsig = cor_2_str(nx, ny)
                tile = self.map.loc[ny, nx]
                if tile == 'O' and not visited.get(nsig, False):
                    q.put((nx, ny, nsig))
                    visited[nsig] = True
            # Down
            if ty < (self.height - 1):
                ny = ty + 1
                nx = tx
                nsig = cor_2_str(nx, ny)
                tile = self.map.loc[ny, nx]
                if tile == 'O' and not visited.get(nsig, False):
                    q.put((nx, ny, nsig))
                    visited[nsig] = True
            # Left
            if tx > 0:
                ny = ty
                nx = tx - 1
                nsig = cor_2_str(nx, ny)
                tile = self.map.loc[ny, nx]
                if tile == 'O' and not visited.get(nsig, False):
                    q.put((nx, ny, nsig))
                    visited[nsig] = True
            # Down
            if tx < (self.width - 1):
                ny = ty
                nx = tx + 1
                nsig = cor_2_str(nx, ny)
                tile = self.map.loc[ny, nx]
                if tile == 'O' and not visited.get(nsig, False):
                    q.put((nx, ny, nsig))
                    visited[nsig] = True

        return True

    def run(self):
        self.pre_calc()

        for idx, (cx, cy) in enumerate(self.coordinates):
            tile = self.map.loc[cy, cx]
            if tile == '#':
                dprint(f'Coor [{idx+1:>3}/{self.coordinates_count}][{cx}x{cy}] --> NOT LAKE')
                # self.results.append(0)
                print(0)
            else:
                sig = cor_2_str(cx, cy)
                if any(self.lakes.loc[sig] > 0):
                    # This coordinate is in a known lake
                    dprint("Answer is known")
                else:
                    # New lake!
                    dprint("Running calculation")
                    self.run_coordinate(cx, cy)

                # Calculate length
                lake = self.lakes.columns[self.lakes.loc[sig] > 0][0]
                lake_size = self.lakes[lake].sum()
                dprint(f'Coor [{idx+1:>3}/{self.coordinates_count}][{cx}x{cy}] --> Lake {lake} Size: {lake_size}')
                # self.results.append(lake_size)
                print(lake_size)


world = World()
d = None
if LOCAL_MODE:
    d = dict()
    d["width"] = 20
    d["height"] = 20
    d["map_raw"] = [
        "########OOOO#OOO####",
        "########OOOOOO######",
        "########OO##########",
        "########OO##########",
        "#######OOO##########",
        "#################O##",
        "#################O##",
        "######OOO######OOO##",
        "######OOOO####OOOO##",
        "##############OOOOO#",
        "##############OOO#OO",
        "##############OOOOOO",
        "#############OOOO###",
        "####################",
        "####################",
        "#####OO#############",
        "####################",
        "####################",
        "####################",
        "##############OOOOO#",
    ]
    d["coordinates_count"] = 10
    d["coordinates"] = [(8, 2), (8, 7), (6, 15), (16, 7), (6, 7), (16, 7), (14, 19), (18, 19), (16, 7), (15, 9)]

world.update(d)
print_dict(world.get_params())

world.run()
