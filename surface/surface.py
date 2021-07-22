import sys
import math
import socket

import numpy as np
import pandas as pd

import pickle as p
from copy import copy
from queue import Queue
from collections import deque
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

        self.lakes = dict()
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

    def floodfil(self, x, y, lake=3):
        mask = np.zeros((self.height + 2, self.width + 2), np.uint8)
        surface_size, qarr1, _, _ = cv2.floodFill(self.map, mask, seedPoint=(x, y), newVal=lake)
        if type(qarr1) == np.ndarray:
            self.map = qarr1
        else:
            self.map = qarr1.get()
        self.lakes[lake] = surface_size

    def floodfil_1(self, x, y, lake):

        if self.get_tile(x, y) == -1:
            self.map[y, x] = lake
            if x > 0:
                self.floodfil_1(x - 1, y, lake)
            if x < self.width - 1:
                self.floodfil_1(x + 1, y, lake)
            if y > 0:
                self.floodfil_1(x, y - 1, lake)
            if y < self.height - 1:
                self.floodfil_1(x, y + 1, lake)

    def floodfil_2(self, x, y, lake):
        q = deque([(x, y)])
        while q:
            x, y = q.pop()
            if self.get_tile(x, y) == -1:
                self.map[y, x] = lake
                if x > 0:
                    q.append((x - 1, y))
                if x < self.width - 1:
                    q.append((x + 1, y))
                if y > 0:
                    q.append((x, y - 1))
                if y < self.height - 1:
                    q.append((x, y + 1))

    def run_coordinate(self, x, y):
        current_lake = self.lakes_count
        self.floodfil_2(x, y, current_lake)
        self.lakes[current_lake] = np.sum(self.map == current_lake)
        self.lakes_count += 1

    def view_map(self):
        l1 = range(self.width)
        l2 = range(self.height)
        self.map = pd.DataFrame(columns=l1, index=l2,
                                data=[list(t) for t in self.map_raw]
                                ).to_numpy()
        self.map[np.where(self.map == '#')] = -5
        self.map[np.where(self.map == 'O')] = -1
        self.map = np.array(self.map, dtype=np.int32)

    def precalc(self):
        l1 = range(self.width)
        l2 = range(self.height)
        l = [cor_2_str(*t) for t in list(itertools.product(l1, l2))]
        self.view_map()
        self.lakes_count = 0

    def get_tile(self, x, y):
        tile = self.map[y, x]
        return tile

    def run(self):
        self.precalc()

        for idx, (cx, cy) in enumerate(self.coordinates):
            tile = self.get_tile(cx, cy)
            if tile == -5:
                dprint(f'Coor [{idx+1:>3}/{self.coordinates_count}][{cx}x{cy}] --> NOT LAKE')
                # self.results.append(0)
                print(0)
            else:
                if tile == -1:
                    # New tile!
                    dprint(
                        f'Coor [{idx+1:>3}/{self.coordinates_count}][{cx}x{cy}] --> Running Lake {self.lakes_count}!.')
                    self.run_coordinate(cx, cy)
                else:
                    dprint(f'Coor [{idx+1:>3}/{self.coordinates_count}][{cx}x{cy}] --> known. Lake {tile}')
                tile = self.get_tile(cx, cy)
                surface = self.lakes.get(tile)
                print(surface)


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
