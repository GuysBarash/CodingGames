import time
from datetime import datetime
import math
import numpy as np
import random
import sys


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


def dprint(s=''):
    print(s, file=sys.stderr)
