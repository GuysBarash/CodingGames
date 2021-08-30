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
    elif action == 'UP_LEFT':
        return x - 1, y - 1
    elif action == 'UP_RIGHT':
        return x + 1, y - 1
    elif action == 'DOWN_LEFT':
        return x - 1, y + 1
    elif action == 'DOWN_RIGHT':
        return x + 1, y + 1
    else:
        raise Exception(f"BAD SIGN: {action}")


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


class World:
    # Map legend:
    # b = black amazon
    # w = white amazon
    # . = available location
    # - = Wall

    @staticmethod
    def translate_coordinates(amazon_src, amazon_move, amazon_shot):
        def _coor2str(s):
            abc = 'abcdefghijklmnopqrstuvwxyz'
            x = abc[s[0]]
            y = 8 - s[1]
            return f'{x}{y}'

        amazon_src_str = _coor2str(amazon_src)
        amazon_move_str = _coor2str(amazon_move)
        amazon_shot_str = _coor2str(amazon_shot)
        res = f'{amazon_src_str}{amazon_move_str}{amazon_shot_str}'
        return res

    @staticmethod
    def switch_sign(c):
        if c == 'w':
            return 'b'
        else:
            return 'w'

    def __init__(self, d=None):
        self.board_size = None
        self.current_color = None
        self.raw = None
        self.last_action = None
        self.actions_count = None
        self.turn = 0

        if d is None:
            self.board_size = int(input())  # height and width of the board
        else:
            self.board_size = d['board_size']

    def get_init_params(self):
        d = dict()
        d['board_size'] = self.board_size
        return d

    def visualize(self):
        pass

    def update(self, d=None):
        read_time = datetime.now()
        if d is None:
            self.current_color = input()  # current color of your pieces ("w" or "b")
            self.raw = [list(input()) for _ in range(self.board_size)]
            self.last_action = input()  # last action made by the opponent ("null" if it's the first turn)
            self.actions_count = int(input())  # number of legal actions
            self.turn += 2
            dprint("Update completed.")
        else:
            self.current_color = d['current_color']
            self.raw = d['raw']
            self.last_action = d['last_action']
            self.actions_count = d['actions_count']
            self.turn = d['turn']

        self.arr = np.array(self.raw)
        read_time = (datetime.now() - read_time).total_seconds()
        dprint(f"Read time: {read_time}")

    def get_params(self):
        d = dict()
        d['current_color'] = self.current_color
        d['raw'] = self.raw
        d['last_action'] = self.last_action
        d['actions_count'] = self.actions_count
        d['board_size'] = self.board_size
        d['turn'] = self.turn
        return d


class State:
    # X is left[0] --> right[N]
    # Y is up[0]   --> down[N]

    # A turn includes:
    # Apply move,
    # Apply arrow,
    # Apply end_turn

    def __init__(self, current_turn, current_color, my_amazons_map, foe_amazons_map, walls_map):
        self.current_color = current_color
        self.current_turn = current_turn

        self.my_amazons_map = my_amazons_map
        self.foe_amazons_map = foe_amazons_map
        self.walls_map = walls_map

        self.arr_size = self.my_amazons_map.shape[0]

        self.last_moves_pience = None
        self.move_done = False
        self.arrow_done = False

    def get_point_of_view(self, x=None, y=None):
        if x is None:
            return self.my_amazons_map - self.foe_amazons_map - self.walls_map
        else:
            r = - self.my_amazons_map - self.foe_amazons_map - self.walls_map
            r[y, x] = 1
            return r

    def get_map(self):
        return 1 * self.my_amazons_map + 2 * self.foe_amazons_map - self.walls_map

    def view(self, arr=None):
        if LOCAL_MODE:
            import matplotlib.pyplot as plt

            if arr is None:
                arr = self.get_map()
            plt.matshow(arr)
            plt.show()
        else:
            pass

    def legal_moves(self, x, y):
        view = self.get_point_of_view(x, y)
        l2r = view[y, :]
        u2d = view[:, x]
        lu2rd = np.diag(view, x - y)

    def utility_by_floodfill(self):
        players = [1, 2]
        player_count = 2
        wall = -1
        xmap = 1 * self.my_amazons_map + 2 * self.foe_amazons_map - self.walls_map

        frontiers = dict()
        utility = {p: 0 for p in players}
        for player in players:
            yarr, xarr = np.where(xmap == player)
            position = [(xarr[i], yarr[i]) for i in range(xarr.shape[0])]
            q = deque()
            [q.append(tp) for tp in position]
            frontiers[player] = q
            utility[player] += len(position)

        while len(frontiers) > 0:
            for player in players:
                new_frontire = deque()
                frontier = frontiers.pop(player, None)
                if frontier is not None:
                    for pos in frontier:
                        for action in ['UP', 'DOWN', 'LEFT', 'RIGHT',
                                       'UP_LEFT', 'UP_RIGHT',
                                       'DOWN_LEFT', 'DOWN_RIGHT',
                                       ]:
                            qx, qy = action_to_xy(pos, action)
                            hit_wall = (qx < 0) or (qx >= self.arr_size) or (qy < 0) or (
                                    qy >= self.arr_size)
                            if not hit_wall:
                                if xmap[qx, qy] == 0:
                                    xmap[qx, qy] = player
                                    utility[player] += 1
                                    new_frontire.append([qx, qy])
                    if len(new_frontire) > 0:
                        frontiers[player] = new_frontire

        r_util = dict()
        r_util['player'] = utility[1]
        r_util['foe'] = utility[2]
        return utility

    def utility_by_single_step_coverage(self):
        players = [1, 2]
        utility = {p: 0 for p in players}
        for player in players:
            xmap = 1 * self.my_amazons_map + 2 * self.foe_amazons_map - self.walls_map
            yarr, xarr = np.where(xmap == player)
            position = [(xarr[i], yarr[i]) for i in range(xarr.shape[0])]
            for p in position:
                for action in ['UP', 'DOWN', 'LEFT', 'RIGHT',
                               'UP_LEFT', 'UP_RIGHT',
                               'DOWN_LEFT', 'DOWN_RIGHT',
                               ]:
                    qx, qy = action_to_xy(p, action)
                    while True:
                        hit_wall = (qx < 0) or (qx >= self.arr_size) or (qy < 0) or (
                                qy >= self.arr_size)
                        if hit_wall:
                            break

                        if xmap[qy, qx] != 0:
                            break

                        xmap[qy, qx] = player
                        qx, qy = action_to_xy([qx, qy], action)
                        utility[player] += 1

        r_util = dict()
        r_util['player'] = utility[1]
        r_util['foe'] = utility[2]
        return utility

    def get_moves(self, pos=None, player='player'):
        # 1 is current player
        # 2 is foe

        if pos is not None:
            xmap = - self.my_amazons_map - self.foe_amazons_map - self.walls_map
            xmap[pos[1], pos[0]] = 1
        else:
            if player == 'player':
                xmap = 1 * self.my_amazons_map - self.foe_amazons_map - self.walls_map
            else:
                xmap = (-1) * self.my_amazons_map + self.foe_amazons_map - self.walls_map

        yarr, xarr = np.where(xmap == 1)
        position = [(xarr[i], yarr[i]) for i in range(xarr.shape[0])]
        moves_count = 0
        moves = dict()
        for idx, p in enumerate(position):
            current_color = -100 - idx
            xmap[p[1], p[0]] = current_color
            for action in ['UP', 'DOWN', 'LEFT', 'RIGHT',
                           'UP_LEFT', 'UP_RIGHT',
                           'DOWN_LEFT', 'DOWN_RIGHT',
                           ]:
                qx, qy = action_to_xy(p, action)
                while True:
                    hit_wall = (qx < 0) or (qx >= self.arr_size) or (qy < 0) or (
                            qy >= self.arr_size)
                    if hit_wall:
                        break

                    if xmap[qy, qx] != 0:
                        break

                    xmap[qy, qx] = current_color
                    moves_count += 1
                    qx, qy = action_to_xy([qx, qy], action)
            xmap[p[1], p[0]] = -3
            yarr, xarr = np.where(xmap == current_color)
            c_position = list()
            for i in range(xarr.shape[0]):
                c_position += [(xarr[i], yarr[i])]
                xmap[yarr[i], xarr[i]] = 0
            moves[p] = c_position

        return moves_count, moves

    def apply_move(self, amazon_from, amazon_to):
        new_state = State(self.current_turn, self.current_color,
                          self.my_amazons_map.copy(), self.foe_amazons_map.copy(), self.walls_map.copy(),
                          )
        new_state.move_done = True
        new_state.arrow_done = False
        new_state.my_amazons_map[amazon_from[1], amazon_from[0]] = 0
        new_state.my_amazons_map[amazon_to[1], amazon_to[0]] = 1
        new_state.last_moves_pience = amazon_to
        return new_state

    def apply_arrow(self, arrow_pos):
        new_state = State(self.current_turn, self.current_color,
                          self.my_amazons_map.copy(), self.foe_amazons_map.copy(), self.walls_map.copy(),
                          )
        new_state.move_done = True
        new_state.arrow_done = True
        new_state.walls_map[arrow_pos[1], arrow_pos[0]] = 1
        new_state.last_moves_pience = None
        return new_state

    def apply_end_turn(self):
        new_state = State(self.current_turn + 1, World.switch_sign(self.current_color),
                          self.foe_amazons_map.copy(), self.my_amazons_map.copy(), self.walls_map.copy(),
                          )

        return new_state


class Agent_greedy_floodfill:
    # Choose the best 1 move
    # Than choose the best arrow

    def __init__(self, world):
        self.world = world
        self.player_color = None

    def observe(self, world):
        self.world = world
        self.player_color = self.world.current_color

        # Arr to current view
        my_amazons_map = (self.world.arr == self.player_color).astype(int)
        foe_amazons_map = (self.world.arr == World.switch_sign(self.player_color)).astype(int)
        walls_map = (self.world.arr == '-').astype(int)
        root_state = State(self.world.turn, self.player_color,
                           my_amazons_map, foe_amazons_map, walls_map
                           )

        best_move = self.find_best_move(root_state)

        return best_move

    def find_best_move(self, root):
        curr = root
        action_count, actions = curr.get_moves(player='player')
        moves_calculated = 0
        mx_utility = -9999
        start_time = datetime.now()
        moves_utility = dict()
        for c_amazon in actions.keys():
            c_moves = actions[c_amazon]
            for c_move in c_moves:
                c_state = curr.apply_move(c_amazon, c_move)
                c_state_player_moves_count, c_state_player_moves = c_state.get_moves(player='player')
                c_state_foe_moves_count, c_state_foe_moves = c_state.get_moves(player='foe')
                moves_calculated += 1
                utililty = c_state_player_moves_count - c_state_foe_moves_count
                moves_utility[(c_amazon, c_move)] = utililty
                mx_utility = max(mx_utility, utililty)
                curr_time = datetime.now() - start_time

        best_src, best_trgt = max(moves_utility, key=moves_utility.get)
        n_curr = curr.apply_move(best_src, best_trgt)
        curr_time_sec = curr_time.total_seconds()
        msg = f'{moves_calculated} moves calculated. Time: {curr_time_sec:>.3f} [{curr_time_sec/moves_calculated:>.5f} per move]]'
        dprint(msg)

        start_time = datetime.now()
        shots_utility = dict()
        shots_calculated = 0
        root_moved_not_shot = root.apply_move(best_src, best_trgt)
        shots_count, shots = root_moved_not_shot.get_moves(pos=best_trgt)
        c_shots = shots[best_trgt]
        for c_shot in c_shots:
            c_state = root_moved_not_shot.apply_arrow(c_shot)
            c_state_player_moves_count, c_state_player_moves = c_state.get_moves(player='player')
            c_state_foe_moves_count, c_state_foe_moves = c_state.get_moves(player='foe')
            shots_calculated += 1
            utililty = c_state_player_moves_count - c_state_foe_moves_count
            shots_utility[c_shot] = utililty
            curr_time = datetime.now() - start_time

        best_shot = max(shots_utility, key=shots_utility.get)
        curr_time_sec = curr_time.total_seconds()
        msg = f'{shots_calculated} shots calculated. Time: {curr_time_sec:>.3f} [{curr_time_sec/shots_calculated:>.5f} per move]]'
        dprint(msg)

        action = World.translate_coordinates(best_src, best_trgt, best_shot)
        return action


class Agent_brute_greedy_floodfill:
    # Choose the best 1 move
    # Than choose the best arrow

    def __init__(self, world):
        self.world = world
        self.player_color = None

    def observe(self, world):
        self.world = world
        self.player_color = self.world.current_color

        # Arr to current view
        my_amazons_map = (self.world.arr == self.player_color).astype(int)
        foe_amazons_map = (self.world.arr == World.switch_sign(self.player_color)).astype(int)
        walls_map = (self.world.arr == '-').astype(int)
        root_state = State(self.world.turn, self.player_color,
                           my_amazons_map, foe_amazons_map, walls_map
                           )

        best_move = self.find_best_move(root_state)

        return best_move

    def find_best_move(self, root):

        def _expand_condition(current_move_util, max_move_util=0):
            if max_move_util > 0:
                return current_move_util > (max_move_util / 2)
            else:
                return current_move_util > 2 * max_move_util

        curr = root
        action_count, actions = curr.get_moves(player='player')
        moves_calculated = 0
        shots_calculated = 0
        mx_move_utility = -9999
        mx_turn_utility = -9999
        start_time = datetime.now()
        moves_utility = dict()
        turn_utility = dict()
        for c_amazon in actions.keys():
            c_moves = actions[c_amazon]
            for c_move in c_moves:
                c_state = curr.apply_move(c_amazon, c_move)
                c_state_player_moves_count, c_state_player_moves = c_state.get_moves(player='player')
                c_state_foe_moves_count, c_state_foe_moves = c_state.get_moves(player='foe')
                moves_calculated += 1
                utililty = c_state_player_moves_count - c_state_foe_moves_count
                moves_utility[(c_amazon, c_move)] = utililty
                mx_move_utility = max(mx_move_utility, utililty)
                if _expand_condition(utililty, mx_move_utility):
                    c_shots_count, c_shots_dict = c_state.get_moves(pos=c_move)
                    c_shots = c_shots_dict[c_move]
                    for c_shot in c_shots:
                        c_c_state = c_state.apply_arrow(c_shot)
                        c_c_state_player_moves_count, c_state_player_moves = c_c_state.get_moves(player='player')
                        c_c_state_foe_moves_count, c_state_foe_moves = c_c_state.get_moves(player='foe')
                        shots_calculated += 1
                        c_c_turn_utililty = c_c_state_player_moves_count - c_c_state_foe_moves_count
                        turn_utility[(c_amazon, c_move, c_shot)] = c_c_turn_utililty
                else:
                    pass

                curr_time = datetime.now() - start_time

        best_src, best_move_trgt, best_shot_trgt = max(turn_utility, key=turn_utility.get)
        optimal_utility = turn_utility[(best_src, best_move_trgt, best_shot_trgt)]
        curr_time_sec = curr_time.total_seconds()
        msg = f'{shots_calculated} shots calculated. Time: {curr_time_sec:>.3f} [{curr_time_sec/shots_calculated:>.5f} per move]]'
        dprint(msg)

        # start_time = datetime.now()
        # shots_utility = dict()
        # shots_calculated = 0
        # root_moved_not_shot = root.apply_move(best_src, best_trgt)
        # shots_count, shots = root_moved_not_shot.get_moves(pos=best_trgt)
        # c_shots = shots[best_trgt]
        # for c_shot in c_shots:
        #     c_state = root_moved_not_shot.apply_arrow(c_shot)
        #     c_state_player_moves_count, c_state_player_moves = c_state.get_moves(player='player')
        #     c_state_foe_moves_count, c_state_foe_moves = c_state.get_moves(player='foe')
        #     shots_calculated += 1
        #     utililty = c_state_player_moves_count - c_state_foe_moves_count
        #     shots_utility[c_shot] = utililty
        #     curr_time = datetime.now() - start_time
        #
        # best_shot = max(shots_utility, key=shots_utility.get)
        # curr_time_sec = curr_time.total_seconds()
        # msg = f'{shots_calculated} shots calculated. Time: {curr_time_sec:>.3f} [{curr_time_sec/shots_calculated:>.5f} per move]]'
        # dprint(msg)

        action = World.translate_coordinates(best_src, best_move_trgt, best_shot_trgt)
        return action


class Agent_floddfil_integrative:
    # Choose the best 1 move
    # Than choose the best arrow

    def __init__(self, world):
        self.world = world
        self.player_color = None

    def observe(self, world):
        self.world = world
        self.player_color = self.world.current_color

        # Arr to current view
        my_amazons_map = (self.world.arr == self.player_color).astype(int)
        foe_amazons_map = (self.world.arr == World.switch_sign(self.player_color)).astype(int)
        walls_map = (self.world.arr == '-').astype(int)
        root_state = State(self.world.turn, self.player_color,
                           my_amazons_map, foe_amazons_map, walls_map
                           )

        best_move = self.find_best_move(root_state)
        return best_move

    def find_best_move(self, root, timecap=0.1):
        def _expand_condition(current_move_util, max_move_util=0):
            if max_move_util > 0:
                return current_move_util >= (max_move_util / 2)
            else:
                return current_move_util >= 2 * max_move_util

        start_time = datetime.now()
        moves_time_cap = 0.05
        shots_time_cap = 0.1

        moves_calculated = 0
        mx_move_utility = -9999
        moves_utility = dict()

        curr = root
        action_count, actions = curr.get_moves(player='player')
        for c_amazon in actions.keys():
            c_moves = actions[c_amazon]
            for c_move in c_moves:
                c_state = curr.apply_move(c_amazon, c_move)
                c_state_player_moves_count, c_state_player_moves = c_state.get_moves(player='player')
                c_state_foe_moves_count, c_state_foe_moves = c_state.get_moves(player='foe')
                moves_calculated += 1
                utililty = c_state_player_moves_count - c_state_foe_moves_count
                mx_move_utility = max(mx_move_utility, utililty)
                moves_utility[(c_amazon, c_move)] = utililty, c_state
            curr_time = (datetime.now() - start_time).total_seconds()
            if curr_time > moves_time_cap:
                dprint(f"Breaking moves search.")
                break

        items = sorted(list(moves_utility.items()), key=lambda t: t[1][0], reverse=True)
        dprint(f"[{curr_time}]Checking {moves_calculated} possible moves. Optimal guess: {items[0][1][0]}")
        dprint(f'Time per move: {curr_time/moves_calculated:>.6f}')

        shots_calculated = 0
        mx_turn_utility = -9999
        turn_utility = dict()
        shots_start_time = datetime.now()

        for idx, ((c_amazon, c_move), (move_utililty, c_move_state)) in enumerate(items):
            if _expand_condition(move_utililty, mx_move_utility):
                c_shots_count, c_shots_dict = c_move_state.get_moves(pos=c_move)
                c_shots = c_shots_dict[c_move]
                for c_shot in c_shots:
                    c_c_state = c_state.apply_arrow(c_shot)
                    c_c_state_player_moves_count, c_state_player_moves = c_c_state.get_moves(player='player')
                    c_c_state_foe_moves_count, c_state_foe_moves = c_c_state.get_moves(player='foe')
                    shots_calculated += 1
                    c_c_turn_utililty = c_c_state_player_moves_count - c_c_state_foe_moves_count
                    turn_utility[(c_amazon, c_move, c_shot)] = c_c_turn_utililty
            else:
                pass

            curr_time = (datetime.now() - start_time).total_seconds()
            if curr_time > shots_time_cap:
                dprint(f"Breaking shots search. [{idx+1}/{len(items)}]")
                break

        best_src, best_move_trgt, best_shot_trgt = max(turn_utility, key=turn_utility.get)
        optimal_utility = turn_utility[(best_src, best_move_trgt, best_shot_trgt)]
        curr_time_sec = (datetime.now() - start_time).total_seconds()
        shots_time_sec = (datetime.now() - shots_start_time).total_seconds()
        msg = f'{shots_calculated} shots calculated. Time: {curr_time_sec:>.3f} [{shots_time_sec/shots_calculated:>.5f} per shot]]'
        msg += f'[Optimal guess: {optimal_utility}]'
        dprint(msg)

        action = World.translate_coordinates(best_src, best_move_trgt, best_shot_trgt)
        return action


d = None
if LOCAL_MODE:
    d = dict()
    d["board_size"] = 8

world = World(d)
print_dict(world.get_init_params())
agent = Agent_floddfil_integrative(world)

while True:
    d = None
    if LOCAL_MODE:
        d = dict()
        d["current_color"] = "w"
        d["raw"] = [['.', '.', '.', '.', '.', '.', 'b', '.'], ['.', '.', '.', '.', '.', '.', '.', '.'],
                    ['.', '.', 'w', '.', 'w', '.', '.', '.'], ['b', '.', '.', '.', '.', '.', '.', '.'],
                    ['.', '.', '.', '.', '.', '.', '.', 'w'], ['.', '.', '.', 'b', '.', 'b', '.', '.'],
                    ['.', '.', '.', '.', '.', '.', '.', '.'], ['.', 'w', '.', '.', '.', '.', '.', '.']]
        d["last_action"] = "null"
        d["actions_count"] = 1279
        d["board_size"] = 8
        d["turn"] = 2

    world.update(d)
    print_dict(world.get_params())

    res = agent.observe(world)
    print(res)

    if LOCAL_MODE:
        dprint("Locally breaking...")
        break
