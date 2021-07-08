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

            self.initial_floor = d['initial_floor']
            self.initial_pos = d['initial_pos']

            self.elevator_ramining = d['elevator_ramining']
            self.clones_remaining = d['clones_remaining']

            self.terminal = d['terminal']
            self.winner = d['winner']

        self.map = None
        self.is_valid = self.direction is not None
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
        self.is_valid = self.direction is not None
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
        if tile in ['_'] and self.elevator_ramining > 0:
            actions += ['ELEVATOR']

        if tile in ['_', '+', 'M']:
            actions += ['BLOCK']
        return actions

    def sig(self):
        s = ''
        s += f'_{self.clone_pos}_{self.clone_floor}_{self.direction}'
        s += f'_{self.clones_remaining}_{self.elevator_ramining}'
        return s


class OptimizedBFSAagent:
    termination_cause = dict()

    def __init__(self):
        self.winner_path = None
        self.winner_path_s = None

        self.map = None
        self.winner_r_c = None

        # BFS
        self.root = None
        self.nodes = None
        self.visited = None
        self.q = None
        self.winner_node = None
        self.nodes_generated = 0

        # Walk
        self.off_course = 0

        # Retry mechanism
        self.bfs_attempts = 0
        self.bfs_time_cap = 0.16
        if LOCAL_MODE:
            self.bfs_time_cap = 9999999
        self.bfs_cap = 500000

        self.duplicates = dict()

    class Node:
        def __init__(self, row, col, direction,
                     elevators, clones, rounds, steps,
                     map, winner_r_c,
                     last_action=None,
                     last_node=None,
                     block_per_floor=dict(),
                     ):
            self.row = row
            self.col = col
            self.direction = direction
            self.elevators = elevators
            self.clones = clones
            self.rounds = rounds
            self.steps = steps

            self.map = map
            self.max_col = map.shape[1] - 1
            self.max_row = map.shape[0] - 1
            self.winner_r_c = winner_r_c
            self.distance_to_target = np.abs(winner_r_c[0] - self.row) + np.abs(winner_r_c[1] - self.col)
            self.priority = - self.steps  # - int(10 * np.abs(winner_r_c[0] - self.row) / self.max_row)

            self.last_action = last_action
            self.last_node_sig = None
            if last_node is not None:
                self.last_node_sig = last_node.sig
            self.actions = None

            self.terminal = False
            self.winner = None
            self.check_terminal()

            self.block_per_floor = block_per_floor

            # self.sig = f'ROW{self.row}_COL{self.col}_DIR{self.direction}_ELV{self.elevators}_CLNS{self.clones}'
            # self.sig = f'ROW{self.row}_COL{self.col}_DIR{self.direction}_ELV{self.elevators}_RND{self.rounds}'
            self.sig = self.make_sig()
            self.p_sig = (-self.priority, (self.sig, self))
            self.diaplay = None

        def check_terminal(self):
            in_bound_r = 0 <= self.row <= self.max_row
            in_bound_c = 0 <= self.col <= self.max_col
            out_of_turns = self.rounds < 0
            not_likely = self.distance_to_target > self.rounds
            if (not in_bound_r) or (not in_bound_c):
                self.terminal = True
                self.winner = False
                OptimizedBFSAagent.termination_cause['BOUND'] = OptimizedBFSAagent.termination_cause.get('BOUND', 0) + 1
            elif out_of_turns:
                self.terminal = True
                self.winner = False
                OptimizedBFSAagent.termination_cause['TURNS'] = OptimizedBFSAagent.termination_cause.get('TURNS', 0) + 1
            elif not_likely:
                self.terminal = True
                self.winner = False
                OptimizedBFSAagent.termination_cause['UNLIKELY'] = OptimizedBFSAagent.termination_cause.get('UNLIKELY',
                                                                                                            0) + 1
            else:
                if self.elevators < 0 or self.clones < 0:
                    self.terminal = True
                    self.winner = False
                    OptimizedBFSAagent.termination_cause['ELEV'] = OptimizedBFSAagent.termination_cause.get(
                        'ELEV', 0) + 1
                else:
                    if (self.row, self.col) == self.winner_r_c:
                        self.terminal = True
                        self.winner = True
                    else:
                        self.terminal = False
                        self.winner = None

        def get_actions(self):
            if self.actions is None:
                actions = ['WAIT']
                if self.row < self.max_row and self.elevators > 0 and self.map.loc[self.row, self.col] != 'M':
                    actions += ['ELEVATOR']
                if self.clones > 0 and self.block_per_floor.get(self.row, 0) == 0:
                    actions += ['BLOCK']
                self.actions = actions
            return self.actions

        def apply(self, action):
            delays = dict()
            delays['WAIT'] = 1
            delays['BLOCK'] = 4
            delays['ELEVATOR'] = 4
            my_delay = delays[action]

            if action == 'WAIT':
                tile = self.map.loc[self.row, self.col]
                if tile in ['_', '+']:
                    return OptimizedBFSAagent.Node(self.row, self.col + self.direction, self.direction,
                                                   self.elevators, self.clones,
                                                   self.rounds - my_delay, self.steps + my_delay,
                                                   self.map, self.winner_r_c,
                                                   block_per_floor=self.block_per_floor,
                                                   last_action=action, last_node=self,
                                                   )
                else:
                    # Elevator
                    return OptimizedBFSAagent.Node(self.row + 1, self.col, self.direction,
                                                   self.elevators, self.clones, self.rounds - my_delay,
                                                   self.steps + my_delay,
                                                   self.map, self.winner_r_c,
                                                   block_per_floor=self.block_per_floor,
                                                   last_action=action, last_node=self,
                                                   )
            elif action == 'BLOCK':
                new_direction = -1 * self.direction
                new_block_per_floor = self.block_per_floor.copy()
                new_block_per_floor[self.row] = 1
                return OptimizedBFSAagent.Node(self.row, self.col + new_direction, new_direction,
                                               self.elevators, self.clones - 1, self.rounds - my_delay,
                                               self.steps + my_delay,
                                               self.map, self.winner_r_c,
                                               block_per_floor=new_block_per_floor,
                                               last_action=action, last_node=self,
                                               )
            elif action == 'ELEVATOR':
                return OptimizedBFSAagent.Node(self.row + 1, self.col, self.direction,
                                               self.elevators - 1, self.clones - 1, self.rounds - my_delay,
                                               self.steps + my_delay,
                                               self.map, self.winner_r_c,
                                               block_per_floor=self.block_per_floor,
                                               last_action=action, last_node=self,
                                               )
            else:
                raise Exception(f"BAD ACTION! {action}")

        def make_sig(self):
            s = f'ROW{self.row}_COL{self.col}_DIR{self.direction}_ELV{self.elevators}'
            return s

        def __str__(self):
            return self.sig

        def __gt__(self, other):
            return self.priority > other.priority
    # return comparison

    def run_bfs(self):
        if self.bfs_attempts == 0:
            direction_map = {'RIGHT': +1, 'LEFT': -1}
            self.root = OptimizedBFSAagent.Node(state.clone_floor, state.clone_pos, direction_map[state.direction],
                                                state.elevator_ramining, state.clones_remaining, state.nb_rounds, 0,
                                                self.map, self.winner_r_c)
            self.pq = PriorityQueue()
            self.nodes = dict()
            self.visited = dict()
            self.pq.put(self.root.p_sig)
            self.nodes[self.root.sig] = self.root

        s_time = datetime.now()
        bfs_rounds = 0
        self.bfs_attempts += 1
        while not self.pq.empty() and self.winner_node is None and bfs_rounds < self.bfs_cap:
            c_runtime = (datetime.now() - s_time).total_seconds()
            if c_runtime > self.bfs_time_cap:
                break
            bfs_rounds += 1
            priority, (node_sig, node) = self.pq.get()
            if node.terminal:
                if node.winner:
                    # Got a win!
                    runtime = datetime.now() - s_time
                    dprint(f"Got a solution! ")
                    dprint(f'[Nodes: {self.nodes_generated}]')
                    dprint(f'[duration: {runtime}]')
                    dprint(f'[Nodes per 100ms: {self.nodes_generated / (10 * runtime.total_seconds())}')
                    self.winner_node = node
                    return True
                else:
                    pass
            else:
                if node_sig in self.visited:
                    compare_a = self.nodes[node.sig]
                    if node.steps < compare_a.steps:
                        self.nodes[node_sig] = node
                        dprint("Better path!")

                else:
                    # Keep exploring
                    self.visited[node_sig] = True
                    self.nodes[node_sig] = node
                    actions = node.get_actions()
                    for action in actions:
                        t_node = node.apply(action)
                        self.nodes_generated += 1
                        self.pq.put(t_node.p_sig)

        runtime = datetime.now() - s_time
        dprint(f"[Nodes: {self.nodes_generated}]Oh no. No solution found.. ")
        dprint(f'[duration: {runtime}]')
        dprint(f'[Nodes per 100ms: {self.nodes_generated / (10 * runtime.total_seconds())}')
        return False

    def build_plan(self):

        def _clean_sig(sig):
            s = '_'.join([t for t in sig.split('_') if re.search('([A-Z]*)', t).group(1) in ['ROW', 'COL', 'DIR']])
            return s

        self.winner_path = dict()
        self.winner_path_s = ''
        current_node = self.winner_node
        prev_node = self.nodes.get(self.winner_node.last_node_sig, None)
        self.winner_path_s = ''

        while prev_node is not None:
            curr_sig = _clean_sig(current_node.sig)
            prev_sig = _clean_sig(prev_node.sig)
            cround = self.root.rounds - prev_node.rounds + 1
            st = f'[ROUND {cround:>3}][{prev_node.rounds:>3}] {prev_sig} --[{current_node.last_action}]--> {curr_sig}\n'
            self.winner_path_s = st + self.winner_path_s
            self.winner_path[prev_sig] = current_node.last_action
            current_node = prev_node
            prev_node = self.nodes.get(current_node.last_node_sig, None)
        dprint("Plan completed.")
        dprint(self.winner_path_s)
        cround = self.root.rounds - self.winner_node.rounds + 1
        dprint(f'[ROUND {cround:>3}] WIN! Sig: <{self.winner_node.sig}>')
        return True

    def run_on_path(self, state):
        direction_map = {'RIGHT': +1, 'LEFT': -1}
        state_sig = f'ROW{state.clone_floor}_COL{state.clone_pos}_DIR{direction_map[state.direction]}'
        if state_sig in self.winner_path:
            self.off_course = 0
            action = self.winner_path.pop(state_sig)
            dprint(f"[{state_sig}] Action [{action}] selected from course.")
        else:
            self.off_course += 1
            dprint(f"[{state_sig}][offcourse: {self.off_course}] No instructions, waiting")
            action = 'WAIT'
        return action

    def observe(self, state):
        if self.map is None:
            dprint("Agent first-time-mount")
            self.map = state.map.copy()
            search_grid = np.where(self.map == 'H')
            row_idx = search_grid[0][0]
            winner_row = self.map.index[row_idx]
            col_idx = search_grid[1][0]
            winner_col = self.map.columns[col_idx]
            self.winner_r_c = winner_row, winner_col
        if not state.is_valid:
            dprint("No agent available, waiting")
            return "WAIT"

        if self.winner_path is None:
            dprint(f"Planning [Attempt {self.bfs_attempts}]")
            bfs_success = self.run_bfs()
            if bfs_success:
                dprint("Plan completed.")
            else:
                dprint("Did not complete on time.")
                return 'WAIT'
            self.build_plan()

        return self.run_on_path(state)


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
    d["nb_floors"] = 10
    d["width"] = 19
    d["nb_rounds"] = 47
    d["exit_floor"] = 9
    d["exit_pos"] = 9
    d["nb_total_clones"] = 41
    d["nb_additional_elevators"] = 0
    d["nb_elevators"] = 17
    d["elevators_map"] = {3: [4, 17], 4: [3, 9], 7: [4, 17], 1: [17, 4], 8: [9], 2: [3, 9], 0: [3, 9], 5: [4, 17],
                          6: [9, 3]}
    d["initial_floor"] = 0
    d["initial_pos"] = 6
    d["elevator_ramining"] = 0
    d["clones_remaining"] = 41
    d["clone_floor"] = 0
    d["clone_pos"] = 6
    d["direction"] = "RIGHT"
    d["terminal"] = True
    d["winner"] = False

state = State(d)
agent = OptimizedBFSAagent()
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
        while agent.winner_path is None:
            action = agent.observe(state)
        print_dict(agent.duplicates)
        break
        state = state.apply(action)
        if state.terminal:
            print(f"Winner: {state.winner}")
            break
    else:
        print(action)
