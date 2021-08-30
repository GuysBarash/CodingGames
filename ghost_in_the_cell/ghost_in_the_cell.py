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


class SimpleAgent:
    # Strategy
    # Look for the nearest plant that can be captured. Send half the force
    def __init__(self, world, current_player=+1):
        self.current_player = current_player
        self.opponent_player = -1 * self.current_player

    def observe(self, world):

        if world.facdf['owner'].unique().shape[0] <= 1:
            return 'WAIT'

        # Choose target
        section_choose_target = True
        if section_choose_target:
            for faction in [self.opponent_player, 0, -99999]:
                if faction != -99999:
                    my_factories_index = world.facdf['owner'].eq(self.current_player)
                    my_units_index = world.unitsdf['owner'].eq(self.current_player)
                    if world.unitsdf['owner'].eq(self.current_player).sum() >= 2:
                        dprint(f"wait for units on the move: {my_units_index.sum()}")
                        return "WAIT"
                    else:
                        my_factories = world.facdf[world.facdf['owner'].eq(self.current_player)].index
                        opponent_factories = world.facdf[world.facdf['owner'].eq(faction)].index
                        if len(opponent_factories) == 0:
                            continue

                        battlesdf = pd.DataFrame(index=my_factories, columns=opponent_factories, data=np.inf)
                        # time_to_battle_df = pd.DataFrame(index=my_factories, columns=opponent_factories)
                        for my_fac in my_factories:
                            my_forces = world.facdf.loc[my_fac, 'cyborgs']
                            for opp_fac in opponent_factories:
                                distance = world.proxdf.loc[my_fac, opp_fac]
                                enemy_forces = world.facdf.loc[opp_fac, 'cyborgs'] + (
                                        distance * world.facdf.loc[opp_fac, 'production'])
                                if my_forces > enemy_forces:
                                    battlesdf.loc[my_fac, opp_fac] = distance

                        if battlesdf.min().min() == np.inf:
                            continue

                        row_battlesdf = battlesdf.min(axis=1)
                        row_battlesdf = row_battlesdf[row_battlesdf < np.inf]
                        selected_attacker = row_battlesdf.idxmin()

                        col_battlesdf = battlesdf.loc[selected_attacker]
                        selected_target = col_battlesdf.idxmin()
                        break

                else:
                    # No optimal move
                    # Attack with biggerst factory, closest target
                    dprint("Grinding opponent")
                    my_factories_index = world.facdf['owner'].eq(self.current_player)
                    candidates = world.facdf.loc[my_factories_index, 'cyborgs'].astype(int)
                    selected_attacker = candidates.idxmax()
                    candidates = world.facdf.loc[~my_factories_index].index
                    candidates = world.proxdf.loc[selected_attacker, candidates].astype(int)
                    selected_target = candidates.idxmin()

        # Send forces
        forces_at_base = int(world.facdf.loc[selected_attacker, 'cyborgs'])
        distance = int(world.proxdf.loc[selected_attacker, selected_target])
        forces_at_targt = int(world.facdf.loc[selected_target, 'cyborgs'] + (
                distance * world.facdf.loc[selected_target, 'production']))

        forces_to_send = int(max(int(forces_at_base * 0.75), forces_at_targt))

        # action
        dprint(
            f"ATTACKING! From [{selected_attacker}][UNITS {forces_at_base}] --> {distance} --> [{selected_target}][UNITS {forces_at_targt}]")
        action = f'MOVE {selected_attacker} {selected_target} {forces_to_send}'
        return action


class GreedyAgent:
    # Strategy
    # always send forces to the nearest factory
    def __init__(self, world, current_player=+1):
        self.current_player = current_player
        self.opponent_player = -1 * self.current_player

    def observe(self, world):

        if world.facdf['owner'].unique().shape[0] <= 1:
            return 'WAIT'

        # Choose target
        commands_to_generate = min(2, world.facdf[world.facdf['owner'] == self.current_player]['owner'].count())
        factories_used = list()
        commands = list()
        for cmd_idx in range(commands_to_generate):
            section_choose_factory = True
            if section_choose_factory:
                optionsfac = world.facdf[world.facdf['owner'] == self.current_player]
                optionsfac = optionsfac[optionsfac['cyborgs'] > 0]

        # Send forces
        forces_at_base = int(world.facdf.loc[selected_attacker, 'cyborgs'])
        distance = int(world.proxdf.loc[selected_attacker, selected_target])
        forces_at_targt = int(world.facdf.loc[selected_target, 'cyborgs'] + (
                distance * world.facdf.loc[selected_target, 'production']))

        forces_to_send = int(max(int(forces_at_base * 0.75), forces_at_targt))

        # action
        dprint(
            f"ATTACKING! From [{selected_attacker}][UNITS {forces_at_base}] --> {distance} --> [{selected_target}][UNITS {forces_at_targt}]")
        action = f'MOVE {selected_attacker} {selected_target} {forces_to_send}'
        return action


def dprint(s=''):
    print(s, file=sys.stderr)


def print_dict(d):
    dprint('d = dict()')
    for k in d.keys():
        v = d[k]
        if type(v) == str:
            dprint(f'd["{k}"] = "{v}"')
        elif type(v) == list:
            s = f'd["{k}"] = [' + '\n'
            for vt in v:
                if type(vt) == str:
                    s += f'\"{vt}\",\n'
                else:
                    s += f'{vt},\n'
            s += '\t]'
            dprint(s)
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

        self.factory_count = -1
        self.entity_count = -1
        self.troops_count = -1

        self.terminal = False
        self.winner = None

        self.details = list()

        self.proxdf = None
        self.facdf = None
        self.unitsdf = None

        if d is None:
            self.factory_count = int(input())  # the number of factories
            self.link_count = int(input())  # the number of links between factories
            self.edges = list()
            self.turn = 0

            self.proxdf = pd.DataFrame(index=range(self.factory_count), columns=range(self.factory_count),
                                       data=np.Inf)  # Index is "FROM", column is "TO"

            self.facdf = pd.DataFrame(index=range(self.factory_count),
                                      columns=['owner', 'entity_id', 'cyborgs', 'production'])

            self.unitsdf = pd.DataFrame(columns=['entity_id', 'src', 'trgt', 'owner', 'cyborgs', 'time'])

            for i in range(self.link_count):
                idx = i
                factory_1, factory_2, distance = [int(j) for j in input().split()]
                self.edges += [(idx, factory_1, factory_2, distance)]
                self.proxdf.loc[factory_1, factory_2] = distance
                self.proxdf.loc[factory_2, factory_1] = distance
        else:
            self.turn = d['turn']
            self.factory_count = d['factory_count']
            self.link_count = d['link_count']
            self.edges = d['edges']
            self.entity_count = d['entity_count']
            self.troops_count = d['troops_count']

            self.proxdf = pd.DataFrame(index=range(self.factory_count), columns=range(self.factory_count),
                                       data=np.Inf)  # Index is "FROM", column is "TO"
            self.facdf = pd.DataFrame(index=range(self.factory_count),
                                      columns=['owner', 'entity_id', 'cyborgs', 'production'])

            for i in range(self.link_count):
                idx, factory_1, factory_2, distance = self.edges[i]
                self.proxdf.loc[factory_1, factory_2] = distance
                self.proxdf.loc[factory_2, factory_1] = distance

        # if DEBUG_MODE:
        #     dprint(f"factories: {self.factory_count}\tRoads: {self.link_count}")
        #     for i in range(self.link_count):
        #         idx, factory_1, factory_2, distance = self.edges[i]
        #         dprint(f"[{idx+1}/{self.link_count}]\t{factory_1} <--{distance}--> {factory_2}")

    def get_params(self):
        d = dict()
        d['turn'] = self.turn
        d['factory_count'] = self.factory_count
        d['troops_count'] = self.troops_count
        d['entity_count'] = self.entity_count
        d['link_count'] = self.link_count
        d['edges'] = self.edges
        d['details'] = self.details
        return d

    def update(self, d=None):
        if d is None:
            self.turn += 1
            self.entity_count = int(input())  # the number of entities (e.g. factories and troops)
            self.troops_count = self.entity_count - self.factory_count
            self.details = list()
            for i in range(self.entity_count):
                inputs = input().split()
                # param 0: entity_id
                # param 1: entity_type
                # param 2: arg_1
                # param 3: arg_2
                # param 4: arg_3
                # param 5: arg_4
                # param 6: arg_5
                t_input = [int(inputs[0]), inputs[1], int(inputs[2]), int(inputs[3]), int(inputs[4]), int(inputs[5]),
                           int(inputs[6])]
                self.details.append(t_input)

        else:
            self.turn = d['turn']
            self.entity_count = d['entity_count']
            self.troops_count = d['troops_count']
            self.details = d['details']

        self.unitsdf = pd.DataFrame(columns=['entity_id', 'src', 'trgt', 'owner', 'cyborgs', 'time'],
                                    index=range(self.troops_count),
                                    )
        troops_pointer = 0
        for idx in range(self.entity_count):
            dtuple = self.details[idx]
            utype = dtuple[1]
            if utype == 'FACTORY':
                entity_id = dtuple[0]
                owner = dtuple[2]
                cyborgs = dtuple[3]
                production = dtuple[4]
                self.facdf.loc[entity_id] = [owner, entity_id, cyborgs, production]

            elif utype == 'TROOP':
                entity_id = dtuple[0]
                owner = dtuple[2]
                src = dtuple[3]
                trgt = dtuple[4]
                unit_count = dtuple[5]
                utime = dtuple[6]
                self.unitsdf.loc[troops_pointer] = [entity_id, src, trgt, owner, unit_count, utime]
                troops_pointer += 1
        self.check_win_condition()

    def apply(self, action_1='WAIT', action_2='WAIT'):
        d = self.get_params()
        nworld = World(d)
        nworld.update(d)

        # Update turn
        nworld.turn += 1

        # Move troops
        nworld.unitsdf['time'] -= 1
        units_in_battle = nworld.unitsdf[nworld.unitsdf['time'] <= 0]
        nworld.unitsdf = nworld.unitsdf[nworld.unitsdf['time'] > 0]

        # Produce bots
        working_factories = nworld.facdf['owner'] != 0
        nworld.facdf.loc[working_factories, 'cyborgs'] += nworld.facdf.loc[working_factories, 'production']

        # Execute commands
        if action_1 == 'WAIT':
            pass
        else:
            raise Exception("TODO")

        if action_2 == 'WAIT':
            pass
        else:
            raise Exception("TODO")

        # BATTLE
        for batteling_factory in units_in_battle['trgt'].unique():
            # Troops fight
            df = units_in_battle[units_in_battle['trgt'] == batteling_factory]
            factory = nworld.facdf.loc[batteling_factory]
            if df['owner'].unique().shape[0] > 1:
                raise Exception("TODO. multiple forces")
            else:
                attack_forces = df['cyborgs'].sum()
                attack_owner = df['owner'].iloc[0]

            if attack_owner == factory['owner']:
                nworld.facdf.loc[batteling_factory, 'cyborgs'] += attack_forces
            else:
                defending_force = factory['cyborgs']
                remaining_forces = defending_force - attack_forces
                if remaining_forces >= 0:
                    nworld.facdf.loc[batteling_factory, 'cyborgs'] = remaining_forces
                else:
                    nworld.facdf.loc[batteling_factory, 'cyborgs'] = - remaining_forces
                    nworld.facdf.loc[batteling_factory, 'owner'] = - attack_owner

        # Winning condition

        return nworld

    def check_win_condition(self):
        self.terminal = False
        self.winner = None

        if self.turn > 200:
            self.terminal = True

            player_minus_1_bots = 0
            player_minus_1_bots += self.facdf.loc[self.facdf['owner'] == -1, 'cyborgs'].sum()
            player_minus_1_bots += self.unitsdf.loc[self.unitsdf['owner'] == -1, 'cyborgs'].sum()

            player_plus_1_bots = 0
            player_plus_1_bots += self.facdf.loc[self.facdf['owner'] == +1, 'cyborgs'].sum()
            player_plus_1_bots += self.unitsdf.loc[self.unitsdf['owner'] == +1, 'cyborgs'].sum()

            if player_minus_1_bots > player_plus_1_bots:
                self.winner = -1
                return self.winner
            else:
                self.winner = +1
                return self.winner
        else:
            player_minus_1_factories = self.facdf.loc[self.facdf['owner'] == -1, 'owner'].count()
            player_plus_1_factories = self.facdf.loc[self.facdf['owner'] == +1, 'owner'].count()
            if player_minus_1_factories <= 0:
                player_minus_1_bots = 0
                player_minus_1_bots += self.facdf.loc[self.facdf['owner'] == -1, 'cyborgs'].sum()
                player_minus_1_bots += self.unitsdf.loc[self.unitsdf['owner'] == -1, 'cyborgs'].sum()
                if player_minus_1_bots <= 0:
                    self.terminal = True
                    self.winner = +1
                    return self.winner
            elif player_plus_1_factories <= 0:
                player_plus_1_bots = 0
                player_plus_1_bots += self.facdf.loc[self.facdf['owner'] == +1, 'cyborgs'].sum()
                player_plus_1_bots += self.unitsdf.loc[self.unitsdf['owner'] == +1, 'cyborgs'].sum()
                if player_plus_1_bots <= 0:
                    self.terminal = True
                    self.winner = -1
                    return self.winner
            else:
                pass
        return self.winner


d = None
if LOCAL_MODE:
    d = dict()
    d["turn"] = 12
    d["factory_count"] = 13
    d["troops_count"] = 3
    d["entity_count"] = 16
    d["link_count"] = 78
    d["edges"] = [
        (0, 0, 1, 1),
        (1, 0, 2, 1),
        (2, 0, 3, 7),
        (3, 0, 4, 7),
        (4, 0, 5, 3),
        (5, 0, 6, 3),
        (6, 0, 7, 4),
        (7, 0, 8, 4),
        (8, 0, 9, 2),
        (9, 0, 10, 2),
        (10, 0, 11, 5),
        (11, 0, 12, 5),
        (12, 1, 2, 4),
        (13, 1, 3, 4),
        (14, 1, 4, 10),
        (15, 1, 5, 1),
        (16, 1, 6, 5),
        (17, 1, 7, 2),
        (18, 1, 8, 7),
        (19, 1, 9, 1),
        (20, 1, 10, 4),
        (21, 1, 11, 3),
        (22, 1, 12, 8),
        (23, 2, 3, 10),
        (24, 2, 4, 4),
        (25, 2, 5, 5),
        (26, 2, 6, 1),
        (27, 2, 7, 7),
        (28, 2, 8, 2),
        (29, 2, 9, 4),
        (30, 2, 10, 1),
        (31, 2, 11, 8),
        (32, 2, 12, 3),
        (33, 3, 4, 15),
        (34, 3, 5, 4),
        (35, 3, 6, 10),
        (36, 3, 7, 3),
        (37, 3, 8, 12),
        (38, 3, 9, 5),
        (39, 3, 10, 9),
        (40, 3, 11, 1),
        (41, 3, 12, 14),
        (42, 4, 5, 10),
        (43, 4, 6, 4),
        (44, 4, 7, 12),
        (45, 4, 8, 3),
        (46, 4, 9, 9),
        (47, 4, 10, 5),
        (48, 4, 11, 14),
        (49, 4, 12, 1),
        (50, 5, 6, 7),
        (51, 5, 7, 4),
        (52, 5, 8, 7),
        (53, 5, 9, 4),
        (54, 5, 10, 4),
        (55, 5, 11, 4),
        (56, 5, 12, 8),
        (57, 6, 7, 7),
        (58, 6, 8, 4),
        (59, 6, 9, 4),
        (60, 6, 10, 4),
        (61, 6, 11, 8),
        (62, 6, 12, 4),
        (63, 7, 8, 10),
        (64, 7, 9, 1),
        (65, 7, 10, 8),
        (66, 7, 11, 1),
        (67, 7, 12, 11),
        (68, 8, 9, 8),
        (69, 8, 10, 1),
        (70, 8, 11, 11),
        (71, 8, 12, 1),
        (72, 9, 10, 6),
        (73, 9, 11, 3),
        (74, 9, 12, 8),
        (75, 10, 11, 8),
        (76, 10, 12, 3),
        (77, 11, 12, 12),
    ]
    d["details"] = [
        [0, 'FACTORY', 1, 3, 0, 0, 0],
        [1, 'FACTORY', 1, 10, 3, 0, 0],
        [2, 'FACTORY', 1, 9, 3, 0, 0],
        [3, 'FACTORY', -1, 22, 3, 0, 0],
        [4, 'FACTORY', -1, 1, 3, 0, 0],
        [5, 'FACTORY', 0, 5, 3, 0, 0],
        [6, 'FACTORY', 1, 4, 3, 0, 0],
        [7, 'FACTORY', 0, 7, 3, 0, 0],
        [8, 'FACTORY', 0, 7, 3, 0, 0],
        [9, 'FACTORY', 1, 11, 2, 0, 0],
        [10, 'FACTORY', -1, 0, 2, 0, 0],
        [11, 'FACTORY', -1, 15, 2, 0, 0],
        [12, 'FACTORY', 0, 2, 2, 0, 0],
        [23, 'TROOP', -1, 1, 4, 1, 1],
        [26, 'TROOP', -1, 1, 4, 1, 2],
        [37, 'TROOP', 1, 6, 10, 15, 4],
    ]
world = World(d)
agent = GreedyAgent(world)

if LOCAL_MODE:
    world.update(d)

# game loop
while not world.terminal:

    if not LOCAL_MODE:
        world.update()

    print_dict(world.get_params())
    action = agent.observe(world)

    # Any valid action, such as "WAIT" or "MOVE source destination cyborgs"
    print(action)

    if LOCAL_MODE:
        world = world.apply(action)
        d = None
