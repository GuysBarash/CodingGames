import sys
import math
import numpy as np
import time
import random
import pandas as pd
from datetime import datetime

DEBUG_MODE = True

import math
import random
from datetime import datetime

import math
import numpy as np
import random
from datetime import datetime

import math
import numpy as np
import random
from datetime import datetime


class MinMax:
    def __init__(self, depth_cap=5, time_cap=None):
        self.depth = depth_cap
        self.time_cap = time_cap
        if time_cap is None:
            self.time_cap = math.inf
        self.time_cap_violation = False
        self.current_runtime = 0
        self.nodes_developed = 0
        self.nodes_terminal = 0

    def search(self, state):
        self.nodes_developed = 0
        self.nodes_terminal = 0
        self.current_start_time = datetime.now()
        self.time_cap_violation = False
        action = self.minimax(state, self.depth, -math.inf, math.inf)[0]
        return action

    def minimax(self, state, depth, alpha, beta):
        valid_locations = state.get_actions()
        last_move = state.opp_previous_action
        if last_move > 0:
            valid_locations_idxs = np.argsort(np.abs(np.array(valid_locations) - last_move))
            valid_locations = np.array(valid_locations)[valid_locations_idxs]
        else:
            random.shuffle(valid_locations)
        is_terminal = state.is_terminal()
        is_maximizer_turn = state.get_current_player() == 1
        self.nodes_developed += 1

        if depth == 0 or is_terminal:
            self.nodes_terminal += 1
            reward = state.get_reward()
            return None, reward
        if self.time_cap_violation:
            reward = state.get_reward()
            return None, reward
        else:
            runtime = (datetime.now() - self.current_start_time).total_seconds()
            self.time_cap_violation = runtime > self.time_cap
            if self.time_cap_violation:
                reward = state.get_reward()
                return None, reward

        if is_maximizer_turn:
            value = -math.inf
            column = random.choice(valid_locations)
            scores = list()
            for col in valid_locations:
                next_state = state.apply_action(col)
                new_score = self.minimax(next_state, depth - 1, alpha, beta)[1]
                scores += [(col, new_score)]
                alpha = max(alpha, value)
                if alpha >= beta:
                    return column, value
            column, value = max(scores, key=lambda x: x[1])
            return column, value - 1

        else:  # Minimizing player
            value = math.inf
            column = random.choice(valid_locations)
            scores = list()
            for col in valid_locations:
                if self.time_cap_violation:
                    new_score = state.get_reward()
                else:
                    next_state = state.apply_action(col)
                    new_score = self.minimax(next_state, depth - 1, alpha, beta)[1]
                scores += [(col, new_score)]
                beta = min(beta, value)
                if alpha >= beta:
                    return column, value

            column, value = min(scores, key=lambda x: x[1])
            return column, value


def dprint(s=''):
    print(s, file=sys.stderr)


def xprint(s=''):
    if DEBUG_MODE:
        print(s)
    else:
        pass


class MinMaxAgent:
    def __init__(self, raw=None):
        dprint("I am a MinMax Agent.")
        # 0 player goes first
        if raw is None:
            raw = [int(i) for i in input().split()]
        self.my_id, self.opp_id = raw
        dprint(f"Original input: {raw}")

        dprint(f"Me: {self.my_id}")
        dprint(f"Opponent: {self.opp_id}")
        if self.my_id == 0:
            s = "Me"
        else:
            s = "Opponent"
        dprint(f"First step: {s}")

        # self.engine = Mcts(iterationLimit=500, depth_cap=10)
        self.engine = MinMax(depth_cap=2, time_cap=9.09)

    def get_action(self, state):
        action = self.engine.search(state)
        # dprint(f'Nodes: {self.engine.nodes_developed}')
        # dprint(f'Time: {datetime.now() - self.engine.current_start_time}')
        # dprint(f'Time violation: {self.engine.time_cap_violation}')
        return action


class GreedyAgent:
    def __init__(self, raw=None):
        dprint("I am a Greedy Agent.")
        # 0 player goes first
        if raw is None:
            raw = [int(i) for i in input().split()]
        self.my_id, self.opp_id = raw
        dprint(f"Original input: {raw}")

        dprint(f"Me: {self.my_id}")
        dprint(f"Opponent: {self.opp_id}")
        if self.my_id == 0:
            s = "Me"
        else:
            s = "Opponent"
        dprint(f"First step: {s}")

    def get_action(self, state):
        current_player = state.current_move
        action = state.check_winning_move(current_player)

        if action is None:
            action = state.check_winning_move(1 - current_player)

        if action is None:
            action = np.random.choice(state.get_actions())

        return action


class State:
    def __init__(self, d):
        self.oopsy_chance = 0.0

        self.player_sign = d['player_sign']
        self.turn_index = d['turn_index']
        self.current_move = d['Current_move']
        self.rows = d['rows']
        self.cols = 9
        self.dirs = ['R', 'L', 'U', 'D',
                     'RU', 'RD',
                     'LU', 'LD',
                     ]

        self.board_rows = dict()
        for i in range(self.rows):
            sig = f'board_row_{i}'
            self.board_rows[sig] = d[sig]

        self.opp_previous_action = d['opp_previous_action']

        # self.df = pd.DataFrame(index=range(self.rows), columns=range(self.cols))
        # for i in self.df.index:
        #     row = list(self.board_rows[f'board_row_{i}'])
        #     self.df.loc[i] = row
        self.arr = np.array([list(s) for s in self.board_rows.values()])

        self.num_valid_actions = d['num_valid_actions']
        if self.num_valid_actions is not None:
            self.actions = dict()
            for i in range(self.num_valid_actions):
                sig = f'action_{i}'
                self.actions[sig] = d[sig]
        else:
            legal_moves = np.where(np.count_nonzero(self.arr == '.', axis=0) > 1)[0]
            self.num_valid_actions = legal_moves.shape[0]
            self.actions = {f'action_{i}': legal_moves[i] for i in range(self.num_valid_actions)}

        self.winner = self.check_winner()
        self.terminal = (self.winner is not None) or ('.' not in np.unique(self.arr))

    def _oops(self):
        return np.random.choice([True, False], p=[self.oopsy_chance, 1 - self.oopsy_chance])

    def zoom(self, r, c, direction='R'):
        if (c < 0) or (c >= self.cols) or (r < 0) or (r >= self.rows):
            raise Exception(f"BAD loc: c{c}xr{r}")

        step_size = 4
        if direction == 'R':
            c_top = min(self.cols, c + step_size)
            c_low = max(0, c - 0)
            points = [(r, ct) for ct in range(c_low, c_top)]

        elif direction == 'L':
            c -= (step_size - 1)
            c_top = min(self.cols, c + step_size)
            c_low = max(0, c)
            points = [(r, ct) for ct in range(c_low, c_top)]
            pass

        elif direction == 'U':
            r -= (step_size - 1)
            r_top = min(self.rows, r + step_size)
            r_low = max(0, r)
            points = [(rt, c) for rt in range(r_low, r_top)]
            pass


        elif direction == 'D':
            r_top = min(self.rows, r + step_size)
            r_low = max(0, r)
            points = [(rt, c) for rt in range(r_low, r_top)]
            pass

        elif direction == 'RU':
            # c -= (step_size - 1)
            r -= (step_size - 1)
            r_top = min(self.rows, r + step_size)
            r_low = max(0, r)
            c_top = min(self.cols, c + step_size)
            c_low = max(0, c)
            points = list(zip(range(r_low, r_top), range(c_low, c_top)))
            pass
        elif direction == 'RD':
            # c -= (step_size - 1)
            # r -= (step_size - 1)
            r_top = min(self.rows, r + step_size)
            r_low = max(0, r)
            c_top = min(self.cols, c + step_size)
            c_low = max(0, c)
            points = list(zip(range(r_low, r_top), range(c_low, c_top)))
            pass
        elif direction == 'LU':
            c -= (step_size - 1)
            r -= (step_size - 1)
            r_top = min(self.rows, r + step_size)
            r_low = max(0, r)
            c_top = min(self.cols, c + step_size)
            c_low = max(0, c)
            points = list(zip(range(r_low, r_top), range(c_low, c_top)))
            pass
        elif direction == 'LD':
            c -= (step_size - 1)
            # r -= (step_size - 1)
            r_top = min(self.rows, r + step_size)
            r_low = max(0, r)
            c_top = min(self.cols, c + step_size)
            c_low = max(0, c)
            points = list(zip(range(r_low, r_top), range(c_low, c_top)))
            pass
        else:
            raise Exception(f"BAD DIR: {direction}")

        l = ['.'] * len(points)
        for idx in range(len(points)):
            r, c = points[idx]
            l[idx] = self.df.loc[r, c]

        return l

    def get_dict(self):
        d = dict()
        d['turn_index'] = self.turn_index
        d['Current_move'] = self.current_move
        d['player_sign'] = self.player_sign
        d['rows'] = self.rows

        for i in range(self.rows):
            sig = f'board_row_{i}'
            d[sig] = self.board_rows[sig]

        d['num_valid_actions'] = self.num_valid_actions
        for i in range(self.num_valid_actions):
            sig = f'action_{i}'
            d[sig] = self.actions[sig]

        d['opp_previous_action'] = self.opp_previous_action
        return d

    def print_dict(self):
        d = self.get_dict()
        dprint()
        dprint('d = dict()')
        for k, v in d.items():
            if type(v) == int:
                dprint(f"d['{k}'] = {v}")
            else:
                dprint(f"d['{k}'] = '{v}'")

    def check_winner(self, player_to_check=['0', '1']):
        dirs = dict()
        dirs['R'] = (0, 0), (0, 4)
        dirs['D'] = (0, 4), (0, 0)
        dirs['RD'] = (0, 4), (0, 4)
        dirs['RU'] = (0, 4), (0, 4)

        # data = self.df.to_numpy()
        for sign in player_to_check:
            for dir_type in ['R', 'D', 'RD', 'RU']:
                data_t = (self.arr == sign).astype(int)
                if dir_type == 'RU':
                    data_t = np.flipud(data_t)
                data_tr = np.pad(data_t, dirs[dir_type], mode='constant', constant_values=0)
                datas_tr = np.zeros(data_t.shape)

                if dir_type == 'R':
                    for i in range(4):
                        datas_tr += data_tr[:, (0 + i):(self.cols + i)]
                elif dir_type == 'D':
                    for i in range(4):
                        datas_tr += data_tr[(0 + i):(self.rows + i), :]

                elif dir_type in ['RD', 'RU']:
                    for i in range(4):
                        datas_tr += data_tr[(0 + i):(self.rows + i), (0 + i):(self.cols + i)]
                else:
                    raise Exception("TODO")

                max_val = datas_tr.max()
                if max_val >= 4:
                    return int(sign)

        return None

    def get_actions(self):
        laction = list(self.actions.values())
        if self.turn_index == 1:
            laction += [-2]

        if len(laction) == 0:
            pass
        return laction

    def apply_action(self, colx):
        legal_actions = self.get_actions()
        if colx == 'STEEL':
            colx = -2
        if colx not in legal_actions:
            raise Exception(f"BAD ACTION SELECTED: {colx}")

        last_action = colx
        if colx == -2:
            colx = self.opp_previous_action
            rowx = np.where(self.arr[:, colx] == '.')[0][-1]
        else:
            rowx = np.where(self.arr[:, colx] == '.')[0][-1]

        d = self.get_dict()
        keys_to_pop = [k for k in d.keys() if 'action_' in k]
        for k in keys_to_pop:
            _ = d.pop(k)

        d['turn_index'] += 1
        d['Current_move'] = 1 - d['Current_move']

        rs = list(d[f'board_row_{rowx}'])
        rs[colx] = str(self.current_move)
        d[f'board_row_{rowx}'] = ''.join(rs)
        d['num_valid_actions'] = None

        d['opp_previous_action'] = last_action
        new_state = State(d)

        # mandatory_moves = new_state.check_winning_move(d['Current_move'])
        # if mandatory_moves is not None:
        #     keys_to_pop = [k for k in d.keys() if 'action_' in k]
        #     for k in keys_to_pop:
        #         _ = d.pop(k)
        #     d['num_valid_actions'] = 1
        #     d[f'action_{0}'] = mandatory_moves
        #     new_state = State(d)

        return new_state

    def is_terminal(self):
        return self.terminal

    def get_current_player(self):
        if self.player_sign == self.current_move:
            # maximizer
            return 1
        else:
            # minimizer
            return -1

    def get_reward(self):
        WIN_REWARD = 100.0
        LOSE_REWARD = -100.0
        TIE_REWARD = -50
        DEPTH_CAP = -1

        if self.is_terminal():
            ending = 'TIE'
            if self.winner == self.player_sign:
                ending = 'WIN'
                # xprint("<<< WIN >> ")
                return WIN_REWARD
            elif self.winner == (1 - self.player_sign):
                # xprint("<<< LOSE >> ")
                ending = 'LOSE'
                return LOSE_REWARD
            else:
                xprint("<<< TIE >> ")
                ending = 'TIE'
                return TIE_REWARD
        else:
            ending = 'DEPTH CAP'
            return DEPTH_CAP

    def check_winning_move(self, player=0):
        legal_moves = self.get_actions()
        for legal_move in legal_moves:
            if legal_move == -2:
                continue
            else:
                colx = legal_move
                potential_col = np.where(self.arr[:, colx] == '.')[0]
                if potential_col.shape[0] == 0:
                    continue
                rowx = np.where(self.arr[:, colx] == '.')[0][-1]
                self.arr[rowx, colx] = str(player)
                winner = self.check_winner([str(player)])
                self.arr[rowx, colx] = '.'
                if winner is not None:
                    wining_moves = colx
                    return wining_moves
                else:
                    pass
        return None


class Board:
    def __init__(self, agent):
        self.h = 7
        self.agent = agent
        self.agent_id = agent.my_id
        self.r = 9
        self.df = pd.DataFrame(columns=range(self.r), index=range(self.h))
        self.current_state = None

    def get_inputs(self, d=None, display=True):
        if d is None:
            d = dict()
            # starts from 0; As the game progresses, first player gets [0,2,4,...] and second player gets [1,3,5,...]
            d['player_sign'] = self.agent_id
            d['turn_index'] = int(input())
            d['Current_move'] = self.agent_id

            # one row of the board (from top to bottom)
            d['rows'] = 7
            for i in range(d['rows']):
                d[f'board_row_{i}'] = input()

            # number of unfilled columns in the board
            d['num_valid_actions'] = int(input())
            for i in range(d['num_valid_actions']):
                # a valid column index into which a chip can be dropped
                d[f'action_{i}'] = int(input())

            # opponent's previous chosen column index (will be -1 for first player in the first turn)
            d['opp_previous_action'] = int(input())

        state = State(d)
        self.current_state = state
        if display:
            state.print_dict()
        return state


# Drop chips in the columns.
# Connect at least 4 of your chips in any direction to win.

# my_id: 0 or 1 (Player 0 plays first)
# opp_id: if your index is 0, this will be 1, and vice versa
agent_inp = None
if DEBUG_MODE:
    agent_inp = [0, 1]
agent = MinMaxAgent(agent_inp)
board = Board(agent)

# game loop
while True:
    if DEBUG_MODE:
        d = dict()
        d['turn_index'] = 12
        d['Current_move'] = 0
        d['player_sign'] = 0
        d['rows'] = 7
        d['board_row_0'] = '.........'
        d['board_row_1'] = '.........'
        d['board_row_2'] = '.........'
        d['board_row_3'] = '...1.....'
        d['board_row_4'] = '...01....'
        d['board_row_5'] = '..011..1.'
        d['board_row_6'] = '.0100..0.'
        d['num_valid_actions'] = 9
        d['action_0'] = 0
        d['action_1'] = 1
        d['action_2'] = 2
        d['action_3'] = 3
        d['action_4'] = 4
        d['action_5'] = 5
        d['action_6'] = 6
        d['action_7'] = 7
        d['action_8'] = 8
        d['opp_previous_action'] = 4
    else:
        d = None

    state = board.get_inputs(d)


    # CHECK

    start_t = datetime.now()
    count = 2000
    from tqdm import tqdm

    for i in tqdm(range(count)):
        # actions = state.get_actions()
        action = agent.get_action(state)
        kstate = state.apply_action(action)

    tot_time = (datetime.now() - start_t).total_seconds()
    time.sleep(0.1)
    print(f"Time per action: {1000*tot_time/count:>.1f} [ms]")
    print(f"Per 10ms: {int(count * 0.01/tot_time)}")
    exit(1)
    action = agent.get_action(state)
    print(action)

    if DEBUG_MODE:
        print("BREAKING, DEBUG MODE")
        break
