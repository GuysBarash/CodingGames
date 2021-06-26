import sys
import math
from datetime import datetime
import numpy as np
import pickle as p
from copy import copy
from collections import deque

DEBUG_MODE = True
LOCAL_MODE = True


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


def hash_arr(arr):
    if arr is not None:
        chunks = list()
        qarr = arr.T
        for i in range(qarr.shape[0]):
            s = ''.join(qarr[i].astype(str))
            chunks.append(s)
    else:
        chunks = ['NO', 'ARRAY', 'HASH']
    return chunks


def unhash_arr(d):
    arr = np.zeros((World.GREED_X, World.GREED_Y), dtype=int)
    qarr = arr.T
    for i in range(2000):
        bt = d.get(f'arr_{i:>03}', None)
        if bt is None:
            break
        else:
            bt = np.array(list(bt), dtype=int)
            qarr[i] = bt
    return qarr.T


def action_to_xy(cxy, action):
    x, y = cxy
    if action == 'UP':
        return x, y - 1
    elif action == 'DOWN':
        return x, y + 1
    elif action == 'LEFT':
        return x - 1, y
    elif action == 'RIGHT':
        return x + 1, y
    else:
        raise Exception(f"BAD SIGN: {action}")


class State:
    GREED_X = 30
    GREED_Y = 20

    def __init__(self, prev_state=None, action=None,
                 arr=None, turn=0, player=1, player_count=2, plater_tail=list(), players_head=list()):
        if prev_state is None:
            # Get state from world
            self.arr = copy(arr)
            self.arr_T = self.arr.T
            self.turn = turn
            self.curr_player = player
            self.player_count = player_count
            self.player = player
            self.player_id = self.player + 1
            self.heads = players_head
            self.tails = plater_tail
            self.is_terminal = False

        else:
            # Get state by applying action
            self.turn = prev_state.turn + 1
            self.player_count = prev_state.player_count
            self.player = (prev_state.player + 1) % self.player_count
            self.curr_player = self.player
            self.player_id = self.player + 1

            # Updates
            self.heads = prev_state.heads.copy()
            current_head = self.heads[prev_state.player]
            new_head = action_to_xy(current_head, action)
            self.heads[prev_state.player] = new_head

            self.tails = prev_state.tails.copy()

            hit_wall = (new_head[0] < 0) or (new_head[0] >= State.GREED_X) or (new_head[1] < 0) or (
                    new_head[1] >= State.GREED_Y)

            if hit_wall:
                self.is_terminal = True
                self.arr = None
                self.arr_T = None
            else:
                empty_slot = prev_state.arr[new_head]
                if empty_slot != 0:
                    # Hit another biker
                    self.is_terminal = True
                    self.arr = None
                    self.arr_T = None
                else:
                    self.is_terminal = False
                    self.arr = prev_state.arr.copy()
                    mark_to_add = prev_state.player_id
                    self.arr[new_head[0], new_head[1]] = mark_to_add
                    self.arr_T = self.arr.T

    def get_reward(self):
        pass

    def get_actions(self, for_player=None):
        if for_player is None:
            for_player = self.curr_player
        curr_pos = self.heads[for_player]

        ret = list()
        for k in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            tx, ty = action_to_xy(curr_pos, k)
            hit_wall = (tx < 0) or (tx >= State.GREED_X) or (ty < 0) or (
                    ty >= State.GREED_Y)
            if hit_wall:
                continue
            empty_slot = self.arr[tx, ty]
            if empty_slot != 0:
                continue
            ret += [k]

        if len(ret) == 0:
            if DEBUG_MODE:
                dprint("NO CHOICE BUT DEATH.")
            return ['UP']
        else:
            return ret

    def check_terminal(self):
        return self.is_terminal

    def apply_action(self, action, player=None):
        if player is None:
            player = self.curr_player

        return State(prev_state=self, action=action, player=player)

    def to_dict(self):
        d = dict()
        d['turn'] = self.turn
        d['player_count'] = self.player_count
        d['player'] = self.player
        d['player_sig'] = self.player + 1
        d['terminal'] = self.is_terminal
        for i in range(self.player_count):
            d[f'p{i}_START_X'], d[f'p{i}_START_Y'] = self.tails[i]
            d[f'p{i}_CURR_X'], d[f'p{i}_CURR_Y'] = self.heads[i]

        for i, chunk in enumerate(hash_arr(self.arr)):
            d[f'arr_{i:>03}'] = chunk
        d['DONE'] = 'V'
        return d


class World:
    GREED_X = 30
    GREED_Y = 20

    def __init__(self):
        self.turn = 0
        self.arr = np.zeros((World.GREED_X, World.GREED_Y), dtype=int)
        self.arr_T = self.arr.T

        self.player_count = None
        self.player = None

        self.state = None

    def update(self, d=None):
        if d is None:
            self.turn += 1
            self.player_count, self.player = [int(i) for i in input().split()]
            self.start_pos = np.zeros((2, self.player_count), dtype=int)
            self.pos = np.zeros((2, self.player_count), dtype=int)
            for i in range(self.player_count):
                # x0: starting X coordinate of lightcycle (or -1)
                # y0: starting Y coordinate of lightcycle (or -1)
                # x1: starting X coordinate of lightcycle (can be the same as X0 if you play before this player)
                # y1: starting Y coordinate of lightcycle (can be the same as Y0 if you play before this player)
                x0, y0, x1, y1 = [int(j) for j in input().split()]
                dprint(f'XY0: <{x0}x{y0}>  XY_NOW<{x1}x{y1}>')
                self.start_pos[i] = x0, y0
                self.pos[i] = x1, y1

            for i in range(self.player_count):
                x1, y1 = self.pos[i]
                self.arr[x1, y1] = i + 1
            self.state = State(arr=self.arr, turn=self.turn, player_count=self.player_count, player=self.player,
                               plater_tail=self.start_pos, players_head=self.pos)

        else:
            self.turn = d['turn']
            self.player_count = d['player_count']
            self.player = d['player']
            self.start_pos = np.zeros((2, self.player_count), dtype=int)
            self.pos = np.zeros((2, self.player_count), dtype=int)
            self.arr = unhash_arr(d)
            for i in range(self.player_count):
                x0, y0 = d[f'p{i}_START_X'], d[f'p{i}_START_Y']
                self.start_pos[i] = x0, y0
                x1, y1 = d[f'p{i}_CURR_X'], d[f'p{i}_CURR_Y']
                self.pos[i] = x1, y1
                self.arr[x0, y0] = i + 1
                self.arr[x1, y1] = i + 1

            self.state = State(arr=self.arr, turn=self.turn, player_count=self.player_count, player=self.player,
                               plater_tail=self.start_pos, players_head=self.pos)

        return self.state

    def display(self):
        print_dict(self.state.to_dict())

        return d


class SimpleAgent:
    def __init__(self, player=0):
        self.player = player

    def set_player(self, player):
        self.player = player

    def observe(self, state):
        actions = state.get_actions(state.player)
        action = actions[0]

        if DEBUG_MODE:
            s = ''
            s += '\n'
            for i in range(state.heads.shape[0]):
                s += f'[P{i+1}][X {state.heads[i,0]}][[Y {state.heads[i,1]}]\n'

            s += '\n'
            s += f'Player: {state.player}' + '\n'
            s += f'Legal actions: {actions}' + '\n'
            for t_action in actions:
                old_pos = state.heads[state.player]
                new_pos = action_to_xy(old_pos, t_action)
                old_sign = state.arr[old_pos[0], old_pos[1]]
                new_sign = state.arr[new_pos[0], new_pos[1]]
                chosen = ''
                if t_action == action:
                    chosen = '<><>'
                s += f'POS: {old_pos}({old_sign})-->{t_action} --> {new_pos} ({new_sign})  {chosen}' + '\n'
            dprint(s)

        return action


class FloodLookAheadAgent:
    def __init__(self, player=0):
        self.player = player

    def set_player(self, player):
        self.player = player

    def observe(self, state):
        self.set_player(state.player)
        current_player = state.player
        other_players = [t for t in range(state.player_count) if t != current_player]

        actions = state.get_actions(self.player)
        actions_utils = dict()
        for action in actions:
            nstate = state.apply_action(action)
            utility = self.get_utility(nstate)
            move_util = utility[current_player] - utility[other_players[0]]
            actions_utils[action] = move_util

        max_util = max(actions_utils.values())
        for taction in actions:
            if actions_utils[taction] == max_util:
                action = taction
                break

        if DEBUG_MODE:
            s = ''
            s += '\n'
            for i in range(state.heads.shape[0]):
                s += f'[P{i+1}][X {state.heads[i,0]}][[Y {state.heads[i,1]}]\n'

            s += '\n'
            s += f'Player: {state.player}' + '\n'
            s += f'Legal actions: {actions}' + '\n'
            for t_action in actions:
                old_pos = state.heads[state.player]
                new_pos = action_to_xy(old_pos, t_action)
                chosen = ''
                if t_action == action:
                    chosen = '<><>'
                s += f'POS: {old_pos}-->{t_action} --> {new_pos} V[{actions_utils[t_action]:>6}] {chosen}' + '\n'
            dprint(s)

        return action

    def get_utility(self, state):
        players = [(self.player + i) % state.player_count for i in range(state.player_count)]

        xmap = state.arr.copy()
        xmap.dtype = int
        xmap[xmap != 0] = -2

        frontiers = dict()
        utility = [0] * state.player_count
        for player in players:
            position = state.heads[player].tolist()
            player_sign = player + 1
            xmap[position[0], position[1]] = player_sign
            q = deque()
            q.append(position)
            frontiers[player] = q
            utility[player] += 1

        while len(frontiers) > 0:
            for player in players:
                player_sign = player + 1
                new_frontire = deque()
                frontier = frontiers.pop(player, None)
                if frontier is not None:
                    for pos in frontier:
                        for action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                            qx, qy = action_to_xy(pos, action)
                            hit_wall = (qx < 0) or (qx >= State.GREED_X) or (qy < 0) or (
                                    qy >= State.GREED_Y)
                            if not hit_wall:
                                if xmap[qx, qy] == 0:
                                    xmap[qx, qy] = player_sign
                                    utility[player] += 1
                                    new_frontire.append([qx, qy])
                    if len(new_frontire) > 0:
                        frontiers[player] = new_frontire
        return utility


world = World()
d = None
agent = FloodLookAheadAgent()
while True:
    if LOCAL_MODE:
        d = dict()
        d["turn"] = 20
        d["player_count"] = 2
        d["player"] = 0
        d["player_sig"] = 1
        d["terminal"] = False
        d["p0_START_X"] = 8
        d["p0_START_Y"] = 1
        d["p0_CURR_X"] = 7
        d["p0_CURR_Y"] = 17
        d["p1_START_X"] = 20
        d["p1_START_Y"] = 18
        d["p1_CURR_X"] = 1
        d["p1_CURR_Y"] = 18
        d["arr_000"] = "000000011000000000000000000000"
        d["arr_001"] = "000000011000000000000000000000"
        d["arr_002"] = "000000010000000000000000000000"
        d["arr_003"] = "000000010000000000000000000000"
        d["arr_004"] = "000000010000000000000000000000"
        d["arr_005"] = "000000010000000000000000000000"
        d["arr_006"] = "000000010000000000000000000000"
        d["arr_007"] = "000000010000000000000000000000"
        d["arr_008"] = "000000010000000000000000000000"
        d["arr_009"] = "000000010000000000000000000000"
        d["arr_010"] = "000000010000000000000000000000"
        d["arr_011"] = "000000010000000000000000000000"
        d["arr_012"] = "000000010000000000000000000000"
        d["arr_013"] = "000000010000000000000000000000"
        d["arr_014"] = "000000010000000000000000000000"
        d["arr_015"] = "000000010000000000000000000000"
        d["arr_016"] = "000000010000000000000000000000"
        d["arr_017"] = "000000010000000000000000000000"
        d["arr_018"] = "022222222222222222222000000000"
        d["arr_019"] = "000000000000000000000000000000"
        d["DONE"] = "V"

    state = world.update(d=d)
    world.display()

    action = agent.observe(state)

    # A single line with UP, DOWN, LEFT or RIGHT
    print(action)
    if LOCAL_MODE:
        print("LOCAL mode. BREAKING.")
        break
