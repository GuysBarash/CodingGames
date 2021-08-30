import sys
import math
import socket

import numpy as np
import pandas as pd

import pickle as p
from copy import copy
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


class Ball:
    def __init__(self, x, y, v, ball_id):
        self.x = x
        self.y = y
        self.ball_id = ball_id
        self.v = v  # Turns remaining
        self.sig = f'{self.x}_{self.y}'


class State:
    # H is hole
    # . is empty space
    # @ is full hole
    # > < ^ v are move signs
    # [0-9]+ are balls

    def __init__(self, arr=None, balls=None, turn=0):
        self.arr = arr
        self.sig = ''.join(self.arr.flatten())
        self.turn = turn
        self.dead_end = False
        if balls is None:
            ypos, xpos = np.where(np.vectorize(str.isnumeric)(arr))
            self.balls = dict()
            self.terminal = True
            for i in range(xpos.shape[0]):
                xt, yt = xpos[i], ypos[i]
                vt = int(arr[yt, xt])
                ball = Ball(xt, yt, vt, ball_id=i)
                self.balls[ball.sig] = ball
                self.winner = False
        else:
            self.balls = balls
            self.winner = len(self.balls) == 0

        # Option A, prioritize by number of balls
        # self.priority = len(self.balls)

        # Option B, prioritize by turn remaining
        self.priority = sum([t.v for t in self.balls.values()])

    def get_priority(self):
        return self.priority

    def get_action(self):
        actions = list()
        # ball_sig, ball = max(self.balls.items(), key=lambda b: b[1].ball_id)
        for ball_sig, ball in self.balls.items():
            a = self._get_action(ball)
            if len(a) == 0:
                self.dead_end = True
                actions = list()
                return actions
            elif len(a) == 1:
                # If on of the balls has only 1 option, you must choose it.
                actions = [(ball_sig, a)]
                return actions
            else:
                actions += [(ball_sig, a)]

        actions = [max(actions, key=lambda ac: self.balls[ac[0]].ball_id)]
        return actions

    def _get_action(self, ball):
        power = ball.v
        h, w = self.arr.shape
        if power > 1:
            actions = list()
            # Down
            if ball.y + power < h:
                preds = [self.arr[ball.y + i, ball.x] in ['.', 'X'] for i in range(1, power)]
                preds += [self.arr[ball.y + power, ball.x] in ['.', 'H']]
                if all(preds):
                    actions += ['DOWN']

            # Up
            if ball.y - power >= 0:
                preds = [self.arr[ball.y - i, ball.x] in ['.', 'X'] for i in range(1, power)]
                preds += [self.arr[ball.y - power, ball.x] in ['.', 'H']]
                if all(preds):
                    actions += ['UP']

            # RIGHT
            if ball.x + power < w:
                preds = [self.arr[ball.y, ball.x + i] in ['.', 'X'] for i in range(1, power)]
                preds += [self.arr[ball.y, ball.x + power] in ['.', 'H']]
                if all(preds):
                    actions += ['RIGHT']

            # LEFT
            if ball.x - power >= 0:
                preds = [self.arr[ball.y, ball.x - i] in ['.', 'X'] for i in range(1, power)]
                preds += [self.arr[ball.y, ball.x - power] in ['.', 'H']]
                if all(preds):
                    actions += ['LEFT']

            return actions

        else:
            actions = list()
            # Down
            if ball.y + power < h:
                preds = self.arr[ball.y + power, ball.x] == 'H'
                if preds:
                    actions += ['DOWN']

            # Up
            if ball.y - power >= 0:
                preds = self.arr[ball.y - power, ball.x] == 'H'
                if preds:
                    actions += ['UP']

            # RIGHT
            if ball.x + power < w:
                preds = self.arr[ball.y, ball.x + power] == 'H'
                if preds:
                    actions += ['RIGHT']

            # LEFT
            if ball.x - power >= 0:
                preds = self.arr[ball.y, ball.x - power] == 'H'
                if preds:
                    actions += ['LEFT']

            return actions

    def apply(self, ball_sig, move):
        nballs = copy(self.balls)
        ball = nballs.pop(ball_sig)
        power = ball.v
        if power == 1:
            narr = self.arr.copy()

            if move == 'DOWN':
                narr[ball.y, ball.x] = 'v'
                narr[ball.y + 1, ball.x] = '@'
            elif move == 'UP':
                narr[ball.y, ball.x] = '^'
                narr[ball.y - 1, ball.x] = '@'
            elif move == 'RIGHT':
                narr[ball.y, ball.x] = '>'
                narr[ball.y, ball.x + 1] = '@'
            elif move == 'LEFT':
                narr[ball.y, ball.x] = '<'
                narr[ball.y, ball.x - 1] = '@'

            n_node = State(narr, nballs, self.turn + 1)
            return n_node

        else:
            narr = self.arr.copy()

            if move == 'DOWN':
                for i in range(power):
                    narr[ball.y + i, ball.x] = 'v'

                ball = Ball(ball.x, ball.y + power, power - 1, ball.ball_id)

                token = narr[ball.y, ball.x]
                if token == '.':
                    narr[ball.y, ball.x] = str(ball.v)
                    nballs[ball.sig] = ball
                elif token == 'H':
                    narr[ball.y, ball.x] = '@'

            elif move == 'UP':
                for i in range(power):
                    narr[ball.y - i, ball.x] = '^'

                ball = Ball(ball.x, ball.y - power, power - 1, ball.ball_id)
                token = narr[ball.y, ball.x]
                if token == '.':
                    narr[ball.y, ball.x] = str(ball.v)
                    nballs[ball.sig] = ball
                elif token == 'H':
                    narr[ball.y, ball.x] = '@'

            elif move == 'RIGHT':
                for i in range(power):
                    narr[ball.y, ball.x + i] = '>'

                ball = Ball(ball.x + power, ball.y, power - 1, ball.ball_id)
                token = narr[ball.y, ball.x]
                if token == '.':
                    narr[ball.y, ball.x] = str(ball.v)
                    nballs[ball.sig] = ball
                elif token == 'H':
                    narr[ball.y, ball.x] = '@'

            elif move == 'LEFT':
                for i in range(power):
                    narr[ball.y, ball.x - i] = '<'

                ball = Ball(ball.x - power, ball.y, power - 1, ball.ball_id)
                token = narr[ball.y, ball.x]
                if token == '.':
                    narr[ball.y, ball.x] = str(ball.v)
                    nballs[ball.sig] = ball
                elif token == 'H':
                    narr[ball.y, ball.x] = '@'

            n_node = State(narr, nballs, self.turn + 1)
            return n_node

    def __lt__(self, nstate):
        return self.priority < nstate.priority


class World:

    def __init__(self):
        self.raw = None
        self.width = None
        self.height = None
        self.root = None

    def update(self, d=None):
        read_time = datetime.now()
        if d is None:
            self.width, self.height = [int(i) for i in input().split()]
            self.raw = [input() for i in range(self.height)]
        else:
            self.width = d['width']
            self.height = d['height']
            self.raw = d['raw']

        self.arr = np.array([list(t) for t in self.raw])
        self.root = State(self.arr)
        self.root.get_action()
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

            actions = node.get_action()
            if len(actions) == 0:
                empty_actions += 1
                continue
            else:
                for ball, moves in actions:
                    for move in moves:
                        n_state = node.apply(ball, move)
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
        j = 3

    def print_winner(self):
        qarr = self.winner_node.arr.copy()

        for bad_s in ['@', 'X']:
            qarr[np.where(qarr == bad_s)] = '.'
        for i in range(qarr.shape[0]):
            s = ''.join(qarr[i, :])
            print(s)


d = None
if LOCAL_MODE:
    d = dict()
    d["width"] = 40
    d["height"] = 10
    d["raw"] = ['5.....X..3...H................HX.....4XH', '......X....XXXXX..............XX..2..HXX',
                '......4H........X..4H.H...3..H....4.....', '.HH.........H5XX.....H................5.',
                'X............XXXX....X.244.2.X..H.5.....', 'X.H..........XXXX.......44...X.........5',
                '..............XX4.......3...H.........3.', '...3......3..X........X....H.H..........',
                '.......HH.....XXXXX.H.X.......XX....H.XX', '3........5....H.H.....X.......HX......XH']

world = World()
world.update(d)
print_dict(world.get_params())

agent = Priority_Agent(world)
agent.run()
agent.print_winner()
