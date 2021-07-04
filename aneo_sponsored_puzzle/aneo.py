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


class World:
    def __init__(self):
        pass

    def update(self, d=None):
        if d is None:
            self.speed = int(input())
            self.light_count = int(input())
            self.lights = dict()

            for i in range(self.light_count):
                distance, duration = [int(j) for j in input().split()]
                self.lights[f'Light_{i}'] = distance, duration
        else:
            self.speed = d['speed']
            self.light_count = d['light_count']
            self.lights = dict()

            for i in range(self.light_count):
                self.lights[f'Light_{i}'] = d[f'Light_{i}']

        self.df = pd.DataFrame(index=range(self.light_count),
                               columns=['distance[m]', 'frequency[sec]'],
                               )
        for i in range(self.light_count):
            distance, duration = self.lights[f'Light_{i}']
            self.df.loc[i] = [distance, duration]

    def get_params(self):
        d = dict()
        d['speed'] = self.speed
        d['light_count'] = self.light_count

        for i in range(self.light_count):
            d[f'Light_{i}'] = self.lights[f'Light_{i}']
        return d


def meter_to_km(m):
    return m / 1000.0


def hour_to_sec(h):
    return h * 60.0 * 60.0


class BruteForceAgent:
    def __init__(self):
        self.world = None

    def verify_speed(self, t_kph):
        df = self.world.df
        distance_in_km = meter_to_km(df['distance[m]'])
        time_to_target_sec = hour_to_sec(distance_in_km / t_kph)
        time_to_target_sec = np.floor(time_to_target_sec.astype(np.float32)).astype(int)
        ticks_to_target = time_to_target_sec // df['frequency[sec]']
        is_target_green = ticks_to_target % 2 == 0
        return all(is_target_green)

    def observe(self, world):
        self.world = world

        max_speed = world.speed
        min_speed = 0
        steps = 1  # world.df['frequency[sec]'].min()
        speeds = np.arange(max_speed, min_speed, - steps)
        for speed_kph in speeds:
            t_speed = speed_kph
            valid = self.verify_speed(t_speed)
            dprint(f"Speed: {t_speed}\t valid: {valid}")
            if valid:
                return t_speed

        return -1


d = None
if LOCAL_MODE:
    d = dict()
    d["speed"] = 80
    d["light_count"] = 4
    d["Light_0"] = (700, 25)
    d["Light_1"] = (2200, 15)
    d["Light_2"] = (3000, 10)
    d["Light_3"] = (4000, 28)

agent = BruteForceAgent()
world = World()
world.update(d)
print_dict(world.get_params())
speed = agent.observe(world)

print(f"{speed}")
