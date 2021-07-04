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


class State:
    def __init__(self, d=None):
        if d is None:
            inps = [int(i) for i in input().split()]
            self.turn = 0
            self.nb_floors = inps[0]
            self.width = inps[1]
            self.nb_rounds = inps[2]
            self.exit_floor = inps[3]
            self.exit_pos = inps[4]
            self.nb_total_clones = inps[5]
            self.nb_additional_elevators = inps[6]
            self.nb_elevators = inps[7]

            self.elevators_map = dict()
            for i in range(self.nb_elevators):
                elevator_floor, elevator_pos = [int(j) for j in input().split()]
                self.elevators_map[elevator_floor] = self.elevators_map.get(elevator_floor, list()) + [elevator_pos]

            self.clone_floor = -1
            self.clone_pos = -1
            self.direction = None

            self.initial_floor = -1
            self.initial_pos = -1

            self.elevator_ramining = self.nb_additional_elevators
            self.clones_remaining = self.nb_total_clones

            self.terminal = False
            self.winner = None

        else:
            self.turn = d['turn']
            self.nb_floors = d['nb_floors']
            self.width = d['width']
            self.nb_rounds = d['nb_floors']
            self.exit_floor = d['exit_floor']
            self.exit_pos = d['exit_pos']
            self.nb_total_clones = d['nb_total_clones']
            self.nb_additional_elevators = d['nb_additional_elevators']
            self.nb_elevators = d['nb_elevators']

            self.elevators_map = d['elevators_map']

            self.clone_floor = d['clone_floor']
            self.clone_pos = d['clone_pos']
            self.direction = d['direction']

            self.initial_floor = d['initial_floor']
            self.initial_pos = d['initial_pos']

            self.elevator_ramining = d['elevator_ramining']
            self.clones_remaining = d['clones_remaining']

            self.terminal = d['terminal']
            self.winner = d['winner']

        self.map = None
        self.check_win_condition()

    def get_pos(self):
        return self.clone_floor, self.clone_pos

    def get_tile(self, floor=None, pos=None):
        if floor is None:
            floor = self.clone_floor
            pos = self.clone_pos
        if 0 <= floor < self.nb_floors and 0 <= pos < self.width:
            if self.map is None:
                self._update_map()
            return self.map.loc[floor, pos]
        else:
            return None

    def get_params(self):
        d = dict()
        d['turn'] = self.turn
        d['nb_floors'] = self.nb_floors
        d['width'] = self.width
        d['nb_rounds'] = self.nb_rounds
        d['exit_floor'] = self.exit_floor
        d['exit_pos'] = self.exit_pos
        d['nb_total_clones'] = self.nb_total_clones
        d['nb_additional_elevators'] = self.nb_additional_elevators
        d['nb_elevators'] = self.nb_elevators

        d['elevators_map'] = copy(self.elevators_map)

        d['initial_floor'] = self.initial_floor
        d['initial_pos'] = self.initial_pos

        d['elevator_ramining'] = self.elevator_ramining
        d['clones_remaining'] = self.clones_remaining

        d['clone_floor'] = self.clone_floor
        d['clone_pos'] = self.clone_pos
        d['direction'] = self.direction

        d['terminal'] = self.terminal
        d['winner'] = self.winner
        return d

    def check_win_condition(self):
        self.terminal = False
        self.winner = None
        cond_a = 0 <= self.clone_floor < self.nb_floors
        cond_b = 0 <= self.clone_pos < self.width
        if not (cond_a and cond_b):
            self.terminal = True
            self.winner = False
            return None

        cond_c = self.elevator_ramining < 0
        cond_d = self.clones_remaining < 0
        if cond_c or cond_d:
            self.terminal = True
            self.winner = False
            return None

        if self.clone_floor == self.exit_floor and self.clone_pos == self.exit_pos:
            self.terminal = True
            self.winner = True
            return None

    def update(self, d=None):
        if d is None:
            self.turn += 1
            inputs = input().split()
            self.clone_floor = int(inputs[0])  # floor of the leading clone
            self.clone_pos = int(inputs[1])  # position of the leading clone on its floor
            self.direction = inputs[2]  # direction of the leading clone: LEFT or RIGHT
            if self.direction == 'NONE':
                self.direction = None

            if self.initial_floor < 0 and self.clone_floor >= 0:
                self.initial_floor = self.clone_floor
                self.initial_pos = self.clone_pos
        else:
            self.turn = d['turn']
            self.nb_floors = d['nb_floors']
            self.width = d['width']
            self.nb_rounds = d['nb_rounds']
            self.exit_floor = d['exit_floor']
            self.exit_pos = d['exit_pos']
            self.nb_total_clones = d['nb_total_clones']
            self.nb_additional_elevators = d['nb_additional_elevators']
            self.nb_elevators = d['nb_elevators']

            self.elevators_map = d['elevators_map']

            self.clone_floor = d['clone_floor']
            self.clone_pos = d['clone_pos']
            self.direction = d['direction']

            self.elevator_ramining = d['elevator_ramining']
            self.clones_remaining = d['clones_remaining']

        self._update_map()
        return self

    def _update_map(self):
        if self.map is None:
            self.map = pd.DataFrame(index=np.arange(self.nb_floors)[::-1],
                                    columns=np.arange(self.width),
                                    data='_'
                                    )
            for k in self.elevators_map.keys():
                pos = self.elevators_map[k]
                self.map.loc[k, pos] = 'M'
            self.map.loc[self.exit_floor, self.exit_pos] = 'H'

            if self.initial_floor >= 0:
                self.map.loc[self.initial_floor, self.initial_pos] = '+'

    def display(self):
        s = ''
        s += f'Turn: {self.turn}' + '\n'
        s += f'Tile at pos: <{self.get_tile()}>' + '\n'
        self._update_map()
        for rowidx in self.map.index:
            row = self.map.loc[rowidx]
            v = row.values.copy()
            if rowidx == self.clone_floor:
                v[self.clone_pos] = '@'
            st = ''.join(v)
            s += st + '\n'

        return s

    def apply(self, action):
        d = self.get_params()
        if action == 'WAIT':
            c_tile = self.get_tile()
            if c_tile == 'M':
                d['clone_floor'] += 1
            else:
                xmove = +1
                if d['direction'] == 'LEFT':
                    xmove = -1
                d['clone_pos'] += xmove

        elif action == 'BLOCK':
            if d['direction'] == 'RIGHT':
                d['direction'] = 'LEFT'
            else:
                d['direction'] = 'RIGHT'
            d['clones_remaining'] -= 1
        elif action == 'ELEVATOR':
            d['elevator_ramining'] -= 1
            d['elevators_map'][d['clone_floor']] = d['elevators_map'].get(d['clone_floor'], list()) + [d['clone_pos']]
            d['clone_floor'] += 1
        else:
            raise Exception("BAD ACTION!")

        nstate = State(d=d)
        return nstate

    def place_elevator(self, floor=None, pos=None):
        if floor is None:
            floor = self.clone_floor
            pos = self.clone_pos
        self.elevator_ramining -= 1
        self.elevators_map[floor] = self.elevators_map.get(floor, list()) + [pos]
        self._update_map()

    def get_actions(self):
        actions = ['WAIT']
        tile = self.get_tile()
        if tile in ['_', '+', 'M']:
            actions += ['BLOCK']
        if tile in ['_'] and self.elevator_ramining > 0:
            actions += ['ELEVATOR']
        return actions

    def sig(self):
        s = ''
        s += f'_{self.clone_pos}_{self.clone_floor}_{self.direction}'
        s += f'_{self.clones_remaining}_{self.elevator_ramining}'
        return s


class BFSagent:
    from collections import deque

    class Node:
        def __init__(self, state, last_move=None, prev_node=None):
            self.state = state
            self.sig = state.sig()
            self.last_move = last_move
            self.prev_node = prev_node
            self.terminal = state.terminal
            self.winner = state.winner

            self.path = list()
            if prev_node is not None:
                self.path = copy(prev_node.path)
            if last_move is not None:
                self.path += [last_move]

    def __init__(self):
        self.current_state = None
        self.root = None
        self.winner_node = None
        self.visited = None
        self.q = None
        self.nodes = None

        self.winner_path = None
        self.nodes_generated = 0

    def run_on_path(self):
        if self.winner_path is None:
            self.winner_path = dict()
            curr_node = self.winner_node
            while curr_node.prev_node is not None:
                pos_sig = f'{curr_node.prev_node.state.clone_pos}_{curr_node.prev_node.state.clone_floor}_{curr_node.prev_node.state.direction}'
                pos_move = curr_node.last_move
                curr_node = curr_node.prev_node
                self.winner_path[pos_sig] = pos_move

        current_sig = f'{self.current_state.clone_pos}_{self.current_state.clone_floor}_{self.current_state.direction}'
        return self.winner_path.get(current_sig, 'WAIT')

    def observe(self, state):
        self.current_state = state
        if self.winner_node is not None:
            dprint("RUNNING ON PATH")
            return self.run_on_path()
        elif state.direction is None:
            dprint("NO AGENT")
            return 'WAIT'
        else:
            dprint("SEARCHING PATH")

            self.root = BFSagent.Node(state)
            self.nodes = dict()
            self.visited, self.q = list(), deque([state.sig()])
            self.nodes[self.root.sig] = self.root
            while len(self.q) > 0 and self.winner_node is None:
                node_sig = self.q.popleft()
                node = self.nodes.get(node_sig)
                self.visited.append(node_sig)
                actions = node.state.get_actions()
                for action in actions:
                    nstate = node.state.apply(action)
                    t_node = BFSagent.Node(nstate, action, node)
                    self.nodes_generated += 1
                    dprint(f"Nodes searched: {self.nodes_generated}")
                    if t_node.state.terminal:
                        if t_node.state.winner:
                            # Got a win!
                            self.winner_node = t_node
                            return self.run_on_path()
                        else:
                            pass
                    else:
                        if t_node.sig in self.visited:
                            pass
                        else:
                            # Keep exploring
                            self.q.append(t_node.sig)
                            self.nodes[t_node.sig] = t_node
                            self.visited.append(t_node.sig)


def print_dict(d):
    dprint('d = dict()')
    for k in d.keys():
        v = d[k]
        if type(v) == str:
            dprint(f'd["{k}"] = "{v}"')
        else:
            dprint(f'd["{k}"] = {v}')
    dprint("")


class SimpleAgent:
    def __init__(self):
        pass

    def observe(self, state):
        floor, pos = state.get_pos()
        actions = state.get_actions()
        dprint(f'LEGAL ACTIONS: {actions}')
        if 'ELEVATOR' in actions:
            return 'ELEVATOR'
        else:
            return 'WAIT'


d = None
if LOCAL_MODE:
    d = dict()
    d["turn"] = 1
    d["nb_floors"] = 13
    d["width"] = 69
    d["nb_rounds"] = 109
    d["exit_floor"] = 11
    d["exit_pos"] = 47
    d["nb_total_clones"] = 100
    d["nb_additional_elevators"] = 4
    d["nb_elevators"] = 36
    d["elevators_map"] = {2: [56, 23, 9, 43, 3, 24], 4: [23, 9], 8: [1, 63, 23, 9], 3: [30, 24, 17, 60], 9: [17, 2],
                          11: [45, 4, 50], 6: [9, 3, 23, 35], 1: [24, 36, 62, 17, 4, 50], 7: [48], 10: [45, 3, 23],
                          5: [4]}
    d["initial_floor"] = 0
    d["initial_pos"] = 6
    d["elevator_ramining"] = 4
    d["clones_remaining"] = 100
    d["clone_floor"] = 0
    d["clone_pos"] = 6
    d["direction"] = "RIGHT"
    d["terminal"] = True
    d["winner"] = False

state = State(d)
agent = BFSagent()
print_dict(state.get_params())

# game loop
while True:

    if LOCAL_MODE:
        pass
    else:
        state.update(d)

    print_dict(state.get_params())
    dprint("")
    dprint(state.display())
    dprint(state.sig())

    action = agent.observe(state)

    if LOCAL_MODE:
        state = state.apply(action)
        if state.terminal:
            print(f"Winner: {state.winner}")
            break
    else:
        print(action)
        state = state.update(d)
