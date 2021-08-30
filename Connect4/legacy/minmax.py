import math
import numpy as np
import random
from datetime import datetime


class MinMax:
    def __init__(self, depth_cap=4, time_cap=None):
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
