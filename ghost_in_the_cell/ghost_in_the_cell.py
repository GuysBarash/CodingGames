import sys
import math
import numpy as np
import pandas as pd

DEBUG_MODE = True

from greedy import Greedy


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


class Factory:
    def __init__(self, **kwargs):
        self.type = 'FACTORY'
        self.entity_id = kwargs['entity_id']
        self.owner = kwargs['owner']
        self.cyborgs = kwargs['cyborgs']
        self.production = kwargs['production']


class Troop:
    def __init__(self, **kwargs):
        self.type = 'TROOP'
        self.entity_id = kwargs['entity_id']
        self.owner = kwargs['owner']
        self.src = kwargs['src']
        self.trgt = kwargs['trgt']
        self.unit_count = kwargs['unit_count']
        self.utime = kwargs['utime']


class World:
    def __init__(self, d=None):
        self.factories = list()
        self.troops = list()
        self.proxdf = None
        self.facdf = None
        self.unitsdf = None

        if d is None:
            self.factory_count = int(input())  # the number of factories
            self.link_count = int(input())  # the number of links between factories
            self.edges = list()
            self.turn = 0
            d = dict()
            d['turn'] = 0
            d['factory_count'] = self.factory_count
            d['link_count'] = self.link_count

            self.proxdf = pd.DataFrame(index=range(d['factory_count']), columns=range(d['factory_count']),
                                       data=np.Inf)  # Index is "FROM", column is "TO"

            self.facdf = pd.DataFrame(index=range(d['factory_count']),
                                      columns=['owner', 'entity_id', 'cyborgs', 'production'])

            self.unitsdf = pd.DataFrame(columns=['entity_id', 'src', 'trgt', 'owner', 'cyborgs', 'time'])

            for i in range(self.link_count):
                idx = i
                factory_1, factory_2, distance = [int(j) for j in input().split()]
                self.edges += [(idx, factory_1, factory_2, distance)]
                self.proxdf.loc[factory_1, factory_2] = distance
                self.proxdf.loc[factory_2, factory_1] = distance
                dprint(f"[{idx+1}/{self.link_count}]\t{factory_1} <--{distance}--> {factory_2}")
            d['edges'] = self.edges
        else:
            self.turn = d['turn']
            self.factory_count = d['factory_count']
            self.link_count = d['link_count']
            self.edges = list()

            self.proxdf = pd.DataFrame(index=range(d['factory_count']), columns=range(d['factory_count']),
                                       data=np.Inf)  # Index is "FROM", column is "TO"
            self.facdf = pd.DataFrame(index=range(d['factory_count']),
                                      columns=['owner', 'entity_id', 'cyborgs', 'production'])

            for i in range(self.link_count):
                idx, factory_1, factory_2, distance = d['edges'][i]
                self.proxdf.loc[factory_1, factory_2] = distance
                self.proxdf.loc[factory_2, factory_1] = distance

        print_dict(d)

        dprint(f"factories: {self.factory_count}\tRoads: {self.link_count}")
        for i in range(self.link_count):
            idx, factory_1, factory_2, distance = d['edges'][i]
            self.edges += [(idx, factory_1, factory_2, distance)]
            dprint(f"[{idx+1}/{self.link_count}]\t{factory_1} <--{distance}--> {factory_2}")

    def step(self, d=None, action=None, display_raw=True, display=False):
        if d is not None:
            # Take step from dict
            d_update = True

        elif action is not None:
            # Action on previous state

            # Add pieces to boars
            p1_move, p2_move = action
            if p1_move == 'WAIT':
                pass
            else:
                pass
            if p2_move == 'WAIT':
                pass
            else:
                pass

            self.unitsdf['time'] -= 1
            zeroidx = self.unitsdf['time'] <= 0

            # Apply attack
            zdf = self.unitsdf[zeroidx]
            for trgt in zdf['trgt'].unique():
                xzdf = zdf[zdf['trgt'] == trgt]
                units = (xzdf['cyborgs'] * xzdf['owner']).sum()
                attack_owner = np.sign(units)
                attack_units = np.abs(units)

                def_owner, def_units = self.facdf.loc[trgt, ['owner', 'cyborgs']]

                if def_owner == attack_owner:
                    self.facdf.loc[trgt, 'cyborgs'] += attack_units
                else:
                    units_remain = def_units - attack_units
                    if units_remain >= 0:
                        self.facdf.loc[trgt, 'cyborgs'] = units_remain
                    else:
                        self.facdf.loc[trgt, ['owner', 'cyborgs']] = attack_owner, - units_remain
            self.unitsdf = self.unitsdf[~zeroidx]

        else:
            # Get from input
            d = dict()
            d['entity_count'] = int(input())  # the number of entities (e.g. factories and troops)
            for i in range(d['entity_count']):
                inputs = input().split()
                d[f'entity_id_X{i}'] = int(inputs[0])
                d[f'entity_type_X{i}'] = inputs[1]
                d[f'arg_1_X{i}'] = int(inputs[2])
                d[f'arg_2_X{i}'] = int(inputs[3])
                d[f'arg_3_X{i}'] = int(inputs[4])
                d[f'arg_4_X{i}'] = int(inputs[5])
                d[f'arg_5_X{i}'] = int(inputs[6])

        troops_count = d['entity_count'] - self.factory_count
        self.unitsdf = pd.DataFrame(columns=['entity_id', 'src', 'trgt', 'owner', 'cyborgs', 'time'],
                                    index=range(troops_count),
                                    )
        troops_pointer = 0

        for idx in range(d['entity_count']):
            utype = d[f'entity_type_X{idx}']
            if utype == 'FACTORY':
                entity_id = d[f'entity_id_X{idx}']
                owner = d[f'arg_1_X{idx}']
                cyborgs = d[f'arg_2_X{idx}']
                production = d[f'arg_3_X{idx}']
                self.facdf.loc[entity_id] = [owner, entity_id, cyborgs, production]

            elif utype == 'TROOP':
                entity_id = d[f'entity_id_X{idx}']
                owner = d[f'arg_1_X{idx}']
                src = d[f'arg_2_X{idx}']
                trgt = d[f'arg_3_X{idx}']
                unit_count = d[f'arg_4_X{idx}']
                utime = d[f'arg_5_X{idx}']
                self.unitsdf.loc[troops_pointer] = [entity_id, src, trgt, owner, unit_count, utime]
                troops_pointer += 1

            else:
                raise Exception(f"BAD TYPE: {utype}")

        if display_raw:
            print_dict(d)
        if display:
            j = 3
        return d


d = None
if DEBUG_MODE:
    d = dict()
    d["turn"] = 0
    d["factory_count"] = 7
    d["link_count"] = 21
    d["edges"] = [(0, 0, 1, 4), (1, 0, 2, 4), (2, 0, 3, 7), (3, 0, 4, 7), (4, 0, 5, 3),
                  (5, 0, 6, 3), (6, 1, 2, 9), (7, 1, 3, 1), (8, 1, 4, 12), (9, 1, 5, 1),
                  (10, 1, 6, 8), (11, 2, 3, 12), (12, 2, 4, 1), (13, 2, 5, 8), (14, 2, 6, 1),
                  (15, 3, 4, 15), (16, 3, 5, 3), (17, 3, 6, 11), (18, 4, 5, 11), (19, 4, 6, 3),
                  (20, 5, 6, 8),
                  ]

world = World(d)
agent = Greedy(world)
foe = Greedy(world)

# game loop
while True:
    d = None
    if DEBUG_MODE:
        d = dict()
        d["entity_count"] = 10
        d["entity_id_X0"] = 0
        d["entity_type_X0"] = "FACTORY"
        d["arg_1_X0"] = 0
        d["arg_2_X0"] = 0
        d["arg_3_X0"] = 0
        d["arg_4_X0"] = 0
        d["arg_5_X0"] = 0
        d["entity_id_X1"] = 1
        d["entity_type_X1"] = "FACTORY"
        d["arg_1_X1"] = 1
        d["arg_2_X1"] = 15
        d["arg_3_X1"] = 0
        d["arg_4_X1"] = 0
        d["arg_5_X1"] = 0
        d["entity_id_X2"] = 2
        d["entity_type_X2"] = "FACTORY"
        d["arg_1_X2"] = -1
        d["arg_2_X2"] = 9
        d["arg_3_X2"] = 0
        d["arg_4_X2"] = 0
        d["arg_5_X2"] = 0
        d["entity_id_X3"] = 3
        d["entity_type_X3"] = "FACTORY"
        d["arg_1_X3"] = -1
        d["arg_2_X3"] = 5
        d["arg_3_X3"] = 1
        d["arg_4_X3"] = 0
        d["arg_5_X3"] = 0
        d["entity_id_X4"] = 4
        d["entity_type_X4"] = "FACTORY"
        d["arg_1_X4"] = -1
        d["arg_2_X4"] = 23
        d["arg_3_X4"] = 1
        d["arg_4_X4"] = 0
        d["arg_5_X4"] = 0
        d["entity_id_X5"] = 5
        d["entity_type_X5"] = "FACTORY"
        d["arg_1_X5"] = -1
        d["arg_2_X5"] = 36
        d["arg_3_X5"] = 1
        d["arg_4_X5"] = 0
        d["arg_5_X5"] = 0
        d["entity_id_X6"] = 6
        d["entity_type_X6"] = "FACTORY"
        d["arg_1_X6"] = -1
        d["arg_2_X6"] = 34
        d["arg_3_X6"] = 1
        d["arg_4_X6"] = 0
        d["arg_5_X6"] = 0
        d["entity_id_X7"] = 18
        d["entity_type_X7"] = "TROOP"
        d["arg_1_X7"] = -1
        d["arg_2_X7"] = 6
        d["arg_3_X7"] = 3
        d["arg_4_X7"] = 2
        d["arg_5_X7"] = 1
        d["entity_id_X8"] = 19
        d["entity_type_X8"] = "TROOP"
        d["arg_1_X8"] = -1
        d["arg_2_X8"] = 6
        d["arg_3_X8"] = 3
        d["arg_4_X8"] = 2
        d["arg_5_X8"] = 6
        d["entity_id_X9"] = 20
        d["entity_type_X9"] = "TROOP"
        d["arg_1_X9"] = -1
        d["arg_2_X9"] = 6
        d["arg_3_X9"] = 0
        d["arg_4_X9"] = 2
        d["arg_5_X9"] = 3

    d = world.step(d)
    a1 = agent.observe(world)
    a2 = foe.observe(world)
    world.step(action=(a1, a2))

    # Any valid action, such as "WAIT" or "MOVE source destination cyborgs"
    print("WAIT")

    if DEBUG_MODE:
        break
