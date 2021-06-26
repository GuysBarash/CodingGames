import sys
import math
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

DEBUG_MODE = False


class Greedy_agent:
    def __init__(self):
        pass

    def prioritize_targets(self, world):
        uninterceptable_z_keys = world.zombies['time_human_to_ash'] > world.zombies['time_z_to_human']
        doomed_humans = world.zombies.loc[uninterceptable_z_keys, 'target_human'].unique()
        intercelptable_humans = np.setdiff1d(np.array(world.humans.index), doomed_humans)
        intercelptable_humans_count = intercelptable_humans.shape[0]

        interception_possible_keys = ~world.zombies['target_human'].astype(int).isin(doomed_humans)
        interception_possible = world.zombies[interception_possible_keys]
        dprint(f"Interceptable zombies: {interception_possible.shape[0]}/{world.zombies.shape[0]}")
        dprint(
            f"Humans salvageable: {world.humans.shape[0] - doomed_humans.shape[0]}/{world.humans.shape[0]}")
        if intercelptable_humans_count <= 0:
            # Cannot intercept, do your best
            zrow = world.zombies.sort_values(by='time_human_to_ash').iloc[0]
            dprint(f"Suicide mission: {zrow['target_human'].astype(int)}")
        elif intercelptable_humans_count <= 1:
            # Single target, protect it
            if interception_possible.shape[0] > 0:
                zrow = interception_possible.sort_values(by='time_z_to_human').iloc[0]
            else:
                # Zombies are gunning for a lost cause
                zrow = world.zombies.sort_values(by='distance_z_to_ash').iloc[0]

            dprint(f"Final stand on: {zrow['target_human'].astype(int)}")
        else:
            # These can be intercepted
            if interception_possible.shape[0] > 0:
                zrow = interception_possible.sort_values(by='time_human_to_ash').iloc[0]
            else:
                # Zombies are gunning for a lost cause
                zrow = world.zombies.sort_values(by='distance_z_to_ash').iloc[0]
            dprint(f"Intercepting: {zrow['target_human'].astype(int)}")

        return zrow.name

    def track(self, world, z_id):
        z = world.zombies.loc[z_id]
        cmd = f"{str(int(z['Xdest']))} {str(int(z['Ydest']))}"
        return cmd

    def observe(self, world):
        target = self.prioritize_targets(world)
        dprint(f"Going after z: <{target}>")
        cmd = self.track(world, target)
        return cmd


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


def distance(x_src, y_src, x_trgt, y_trgt):
    xd = x_trgt - x_src
    xd = np.square(xd.astype(np.float64))
    yd = y_trgt - y_src
    yd = np.square(yd.astype(np.float64))
    r = xd + yd
    r = np.sqrt(r)
    return r


def direction(x_src, y_src, x_trgt, y_trgt):
    r = (x_trgt - x_src) / (y_trgt - y_src)
    dg = np.arctan(r)
    return dg


def velocity(X, Y):
    return Y - X


def time_to_taget(distance, speed):
    t = distance / speed
    t = np.floor(t).astype(int)
    return t


# Save humans, destroy zombies!
class World:
    def __init__(self):
        self.humans = None
        self.zombies = None
        self.ash = None
        self.turn = 0

        self.distances = None

        self.zombie_speed = 400
        self.zombie_range = 400
        self.ash_speed = 1000
        self.ash_range = 2000

        self.speed_ratio = self.zombie_speed / self.ash_speed

    def update(self, d=None):
        if d is None:
            self.turn += 1
            self.ash = [int(i) for i in input().split()]
            human_count = int(input())
            self.humans = pd.DataFrame(index=range(human_count), columns=['X', 'Y', 'id'])
            for i in range(human_count):
                human_id, human_x, human_y = [int(j) for j in input().split()]
                self.humans.loc[i] = [human_x, human_y, human_id]

            zombie_count = int(input())
            self.zombies = pd.DataFrame(index=range(human_count), columns=['X', 'Y', 'Xdest', 'Ydest', 'id'])
            for i in range(zombie_count):
                zombie_id, zombie_x, zombie_y, zombie_xnext, zombie_ynext = [int(j) for j in input().split()]
                self.zombies.loc[i] = [zombie_x, zombie_y, zombie_xnext, zombie_ynext, zombie_id]
        else:
            self.turn = d['turn']
            self.ash = d['ash_x'], d['ash_y']

            human_count = d['human_count']
            self.humans = pd.DataFrame(index=range(human_count), columns=['X', 'Y', 'id'])
            for i in range(human_count):
                human_id, human_x, human_y = d[f'human_{i}_id'], d[f'human_{i}_X'], d[f'human_{i}_Y']
                self.humans.loc[i] = [human_x, human_y, human_id]

            zombie_count = d['zombie_count']
            self.zombies = pd.DataFrame(index=range(human_count), columns=['X', 'Y', 'Xdest', 'Ydest', 'id'])
            for i in range(zombie_count):
                self.zombies.loc[i] = [d[f'zombie_{i}_X'], d[f'zombie_{i}_Y'],
                                       d[f'zombie_{i}_Xdest'], d[f'zombie_{i}_Ydest'],
                                       d[f'zombie_{i}_id']]

        self.zombies = self.zombies[~self.zombies.isna().any(axis=1)]
        self.zombies['distance_z_to_ash'] = distance(self.zombies['X'], self.zombies['Y'],
                                                     self.ash[0], self.ash[1],
                                                     )

        # Calculate human interception time
        self.humans['distance_human_to_ash'] = distance(self.humans['X'], self.humans['Y'],
                                                        self.ash[0], self.ash[1],
                                                        )
        self.humans['time_human_to_ash'] = np.floor(self.humans['distance_human_to_ash'] / self.ash_speed)

        # Calculate zombies targets, distance and time to target
        self.zombies['distance_z_to_ash'] = distance(self.zombies['X'], self.zombies['Y'],
                                                     self.ash[0], self.ash[1],
                                                     )
        distances = cdist(self.zombies[['X', 'Y']], self.humans[['X', 'Y']])
        self.distances = distances
        human_target = distances.argmin(axis=1)
        distance_to_human = distances.min(axis=1)
        self.zombies['target_human'] = human_target
        self.zombies['distance_z_to_human'] = distance_to_human
        self.zombies['time_z_to_human'] = np.floor(self.zombies['distance_z_to_human'] / self.zombie_speed)
        self.zombies = self.zombies.join(self.humans, on='target_human', rsuffix='_Human')

        # Calculate interception position and time
        self.zombies['Interception position_X'] = self.zombies['Xdest']
        self.zombies['Interception position_X'] = self.zombies['Ydest']

        # Znext = self.zombies[['Xdest', 'Ydest']].magic_values
        # ZV = velocity(Z, Znext)
        #
        # time_horizon = 30
        # steps_vector = np.array([range(time_horizon), range(time_horizon)]).T
        # ZV3d = ZV.reshape((2, 1, ZV.shape[1]))
        # future_positions = Z.T + steps_vector * ZV.T
        # j = 3

    def get_d(self):
        d = dict()
        d['turn'] = self.turn
        d['ash_x'], d['ash_y'] = self.ash
        d['human_count'] = self.humans.shape[0]
        for i in range(self.humans.shape[0]):
            d[f'human_{i}_X'], d[f'human_{i}_Y'], d[f'human_{i}_id'] = self.humans.loc[i, ['X', 'Y', 'id']]

        d['zombie_count'] = self.zombies.shape[0]
        for i in range(self.zombies.shape[0]):
            zrow = self.zombies.loc[i]
            d[f'zombie_{i}_X'], d[f'zombie_{i}_Y'], d[f'zombie_{i}_id'] = zrow[['X', 'Y', 'id']]
            d[f'zombie_{i}_Xdest'], d[f'zombie_{i}_Ydest'] = zrow[['Xdest', 'Ydest']]

        return d


world = World()
agent = Greedy_agent()
# game loop
while True:
    d = None
    world.update(d)
    if DEBUG_MODE:
        print_dict(world.get_d())
    action = agent.observe(world)

    # Your destination coordinates
    print(action)
