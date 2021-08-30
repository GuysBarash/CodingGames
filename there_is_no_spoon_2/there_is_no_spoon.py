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


class Node:
    @staticmethod
    def connect_nodes(node_1, node_2, action):
        connection = 1
        if '2' in action:
            connection = 2

        node_1.connect(node_2.sig, connection, True)
        node_2.connect(node_1.sig, connection, False)

    @staticmethod
    def are_connected(node_1, node_2):
        n1_cons = node_1.connections
        n2_cons = node_2.connections
        if len(n1_cons) == 0 or len(n2_cons) == 0:
            return False
        else:
            return node_1.connections_sig.get(node_2.sig, False)

    def __init__(self, x, y, v, connections=None, connections_sig=None):
        self.x = x
        self.y = y
        self.v = v
        self.priority = (10000 * (self.v)) + (100 * self.y) + self.x
        self.sig = f'{x}_{y}'
        if connections is None:
            self.connections_sig = dict()
            self.connections = list()
            self.connections_count = 0
            self.slots = self.v - self.connections_count
        else:
            self.connections = connections
            self.connections_sig = connections_sig
            self.connections_count = len(connections)
            self.slots = self.v - self.connections_count

        self.done = self.slots == 0
        if self.done:
            self.priority = -1

    def connect(self, node_sig, connection=1, printable=True):
        self.connections.append((node_sig, connection, printable))
        self.connections_sig[node_sig] = True
        self.connections_count += connection
        self.slots -= connection
        self.done = self.slots <= 0
        if self.done:
            self.priority = -1


class State:
    def __init__(self, arr, nodes=None, step=0):
        self.arr = arr
        self.h, self.w = self.arr.shape
        self.sig = ''.join(self.arr.flatten())
        self.nodes = nodes
        self.step = step
        if self.nodes is None:
            self.nodes = dict()
            self.scan_nodes()
        self.winner = all([v.done for v in self.nodes.values()])
        self.pointer_node_sig, self.pointer_node = max(self.nodes.items(), key=lambda n: n[1].priority)
        self.priority = self.pointer_node.v
        # if self.winner:
        #     j = 3

    def scan_nodes(self):
        for yt in range(self.arr.shape[0]):
            for xt in range(self.arr.shape[1]):
                token = self.arr[yt, xt]
                if token.isnumeric():
                    node = Node(xt, yt, int(token))
                    self.nodes[node.sig] = node

                else:
                    pass

    def get_actions(self):
        # UP, UP2
        # DOWN, DOWN2
        # LEFT, LEFT2
        # RIGHT, RIGHT2

        node_sig, node = self.pointer_node_sig, self.pointer_node
        actions = list()
        mx_connections = 0

        # Up
        for i in range(1, 35):
            t_x = node.x
            t_y = node.y - i
            if t_y < 0:
                break
            token = self.arr[t_y, t_x]
            if token == '.':
                pass
            elif token.isnumeric():
                t_node = self.nodes[f'{t_x}_{t_y}']
                if Node.are_connected(node, t_node):
                    break
                elif t_node.slots == 0:
                    break
                elif t_node.slots == 1 or node.slots == 1:
                    actions += [(node_sig, t_node.sig, 'UP')]
                    mx_connections += 1
                    break
                else:
                    actions += [(node_sig, t_node.sig, 'UP'), (node_sig, t_node.sig, 'UP2')]
                    mx_connections += 2
                    break
            else:
                # Blocked
                break

        # Down
        for i in range(1, 35):
            t_x = node.x
            t_y = node.y + i
            if t_y >= self.h:
                break
            token = self.arr[t_y, t_x]
            if token == '.':
                pass
            elif token.isnumeric():
                t_node = self.nodes[f'{t_x}_{t_y}']
                if Node.are_connected(node, t_node):
                    break
                if t_node.slots == 0:
                    break
                elif t_node.slots == 1 or node.slots == 1:
                    actions += [(node_sig, t_node.sig, 'DOWN')]
                    mx_connections += 1
                    break
                else:
                    actions += [(node_sig, t_node.sig, 'DOWN'), (node_sig, t_node.sig, 'DOWN2')]
                    mx_connections += 2
                    break
            else:
                # Blocked
                break

        # RIGHT
        for i in range(1, 35):
            t_x = node.x + i
            t_y = node.y
            if t_x >= self.w:
                break
            token = self.arr[t_y, t_x]
            if token == '.':
                pass
            elif token.isnumeric():
                t_node = self.nodes[f'{t_x}_{t_y}']
                if t_node.slots == 0:
                    break
                elif Node.are_connected(node, t_node):
                    break
                elif t_node.slots == 1 or node.slots == 1:
                    actions += [(node_sig, t_node.sig, 'RIGHT')]
                    mx_connections += 1
                    break
                else:
                    actions += [(node_sig, t_node.sig, 'RIGHT'), (node_sig, t_node.sig, 'RIGHT2')]
                    mx_connections += 2
                    break
            else:
                # Blocked
                break

        # LEFT
        for i in range(1, 35):
            t_x = node.x - i
            t_y = node.y
            if t_x < 0:
                break
            token = self.arr[t_y, t_x]
            if token == '.':
                pass
            elif token.isnumeric():
                t_node = self.nodes[f'{t_x}_{t_y}']
                if Node.are_connected(node, t_node):
                    break
                if t_node.slots == 0:
                    break
                elif t_node.slots == 1 or node.slots == 1:
                    actions += [(node_sig, t_node.sig, 'LEFT')]
                    mx_connections += 1
                    break
                else:
                    actions += [(node_sig, t_node.sig, 'LEFT'), (node_sig, t_node.sig, 'LEFT2')]
                    mx_connections += 2
                    break
            else:
                # Blocked
                break

        if mx_connections < node.slots:
            actions = list()
        return actions

    def apply(self, action):
        src_node_sig, trgt_node_sig, action_sig = action
        n_nodes = deepcopy(self.nodes)
        src_node = n_nodes.pop(src_node_sig)
        trgt_node = n_nodes.pop(trgt_node_sig)
        Node.connect_nodes(src_node, trgt_node, action_sig)

        narr = self.arr.copy()
        narr[src_node.y, src_node.x] = src_node.slots
        narr[trgt_node.y, trgt_node.x] = trgt_node.slots

        double_mark = '2' in action_sig
        if 'UP' in action_sig:
            for yi in range(src_node.y - 1, trgt_node.y, -1):
                if double_mark:
                    narr[yi, src_node.x] = 'H'
                else:
                    narr[yi, src_node.x] = '|'
        elif 'DOWN' in action_sig:
            for yi in range(src_node.y + 1, trgt_node.y):
                if double_mark:
                    narr[yi, src_node.x] = 'H'
                else:
                    narr[yi, src_node.x] = '|'
        elif 'LEFT' in action_sig:
            for xi in range(src_node.x - 1, trgt_node.x, -1):
                if double_mark:
                    narr[src_node.y, xi] = '='
                else:
                    narr[src_node.y, xi] = '-'
        else:  # RIGHT
            for xi in range(src_node.x + 1, trgt_node.x):
                if double_mark:
                    narr[src_node.y, xi] = '='
                else:
                    narr[src_node.y, xi] = '-'

        n_nodes[src_node_sig] = src_node
        n_nodes[trgt_node_sig] = trgt_node
        n_state = State(narr, n_nodes, self.step + 1)
        return n_state

    def __lt__(self, nstate):
        return self.priority < nstate.priority


class World:

    def __init__(self):
        self.raw = None
        self.width = None
        self.height = None
        self.arr = None

        self.root = None

    def update(self, d=None):
        read_time = datetime.now()
        if d is None:
            self.width = int(input())  # the number of cells on the X axis
            self.height = int(input())  # the number of cells on the Y axis
            self.raw = [input() for _ in range(self.height)]
        else:
            self.width = d['width']
            self.height = d['height']
            self.raw = d['raw']

        self.arr = np.array([list(t) for t in self.raw])
        self.root = State(self.arr)
        read_time = (datetime.now() - read_time).total_seconds()
        dprint(f"Read time: {read_time}")

    def get_params(self):
        d = dict()
        d['width'] = self.width
        d['height'] = self.height
        d['raw'] = self.raw
        return d


class Priority_Agent:
    def __init__(self, world):
        self.world = world
        self.root = world.root
        self.winner_node = None

    def run(self):
        self.winner_node = None
        visited = dict()
        skipped = 0
        empty_actions = 0
        not_skipped = 0
        pq = PriorityQueue()
        pq.put((self.root.priority, self.root))
        while (not pq.empty()) and (self.winner_node is None):
            node = pq.get()[1]
            if visited.get(node.sig, False):
                skipped += 1
                continue
            else:
                not_skipped += 1
                visited[node.sig] = True

            # Displaying the path having lowest cost
            if node.winner:
                break

            actions = node.get_actions()
            if len(actions) == 0:
                empty_actions += 1
                continue
            else:
                for action in actions:
                    n_state = node.apply(action)
                    if n_state.winner:
                        self.winner_node = n_state
                        dprint("Got a Winner!")
                        dprint(f"Total Skipped: {skipped}")
                        dprint(f"Total Opened: {not_skipped}")
                        dprint(f"Useless states: {empty_actions}")
                        break
                    else:
                        pq.put((n_state.priority, n_state))
                if self.winner_node is not None:
                    break

    def print_winner(self):
        # Filter links
        nodes = self.winner_node.nodes.values()
        actions = list()
        for node in nodes:
            s_x = node.x
            s_y = node.y
            for trgt_sig, power, valid in node.connections:
                if valid:
                    t_xy = [int(tt) for tt in trgt_sig.split('_')]
                    t_x, t_y = t_xy
                    # actions.append(f'{s_y} {s_x} {t_y} {t_x} {power}')
                    actions.append(f'{s_x} {s_y} {t_x} {t_y} {power}')
        for action in actions:
            print(action)


d = None
if LOCAL_MODE:
    d = dict()
    d["width"] = 4
    d["height"] = 4
    d["raw"] = ['25.1', '47.4', '..1.', '3344']

world = World()
world.update(d)
print_dict(world.get_params())

agent = Priority_Agent(world)
agent.run()
agent.print_winner()
