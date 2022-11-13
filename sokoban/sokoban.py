# A state machine for the Sokoban game.
# The state is a tuple of the form (player, boxes, goals, walls).

# Legend:
#   Y = player
#   O = box
#   X = goal
#   # = wall
#   _ = empty space
#   @ = box on goal
#   T = player on goal

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import deque
from datetime import datetime
from PIL import Image


class Sokoban:
    def __init__(self, step_cap=10000):
        self.step_cap = step_cap
        self.board = None
        self.player_pos = None
        self.terminal = False
        self.win = False
        self.history = dict()

    def start(self, board):
        self.step_count = 0
        self.board = np.array(board)
        self.board_size = self.board.shape
        self.player_pos = self._find_player()
        self.terminal = False
        self.win = False
        print(self)

    ############################
    # interface functions
    ############################

    def get_actions(self):
        actions = []
        if self.terminal:
            return actions
        x, y = self.player_pos
        if x > 0:
            if self.board[x - 1][y] in ['_', 'X']:
                actions.append('u')
            elif self.board[x - 1][y] in ['O', '@']:
                if x > 1:
                    if self.board[x - 2][y] in ['_', 'X']:
                        actions.append('U')
        if x < self.board_size[0] - 1:
            if self.board[x + 1][y] in ['_', 'X']:
                actions.append('d')
            elif self.board[x + 1][y] in ['O', '@']:
                if x < self.board_size[0] - 2:
                    if self.board[x + 2][y] in ['_', 'X']:
                        actions.append('D')
        if y > 0:
            if self.board[x][y - 1] in ['_', 'X']:
                actions.append('l')
            elif self.board[x][y - 1] in ['O', '@']:
                if y > 1:
                    if self.board[x][y - 2] in ['_', 'X']:
                        actions.append('L')
        if y < self.board_size[1] - 1:
            if self.board[x][y + 1] in ['_', 'X']:
                actions.append('r')
            elif self.board[x][y + 1] in ['O', '@']:
                if y < self.board_size[1] - 2:
                    if self.board[x][y + 2] in ['_', 'X']:
                        actions.append('R')

        return actions

    def apply_action(self, action):

        n_state = deepcopy(self)
        n_state.step_count += 1
        n_state.history[n_state.step_count] = action
        if n_state.step_count > n_state.step_cap:
            n_state.terminal = True

        x, y = n_state.player_pos
        curr_symbol = n_state.board[x][y]

        box_x, box_y = None, None
        new_box_x, new_box_y = None, None
        curr_box_symbol = None

        if action == 'u':
            new_x, new_y = x - 1, y
        elif action == 'd':
            new_x, new_y = x + 1, y
        elif action == 'l':
            new_x, new_y = x, y - 1
        elif action == 'r':
            new_x, new_y = x, y + 1
        elif action == 'U':
            new_x, new_y = x - 1, y
            box_x, box_y = x - 1, y
            new_box_x, new_box_y = x - 2, y
        elif action == 'D':
            new_x, new_y = x + 1, y
            box_x, box_y = x + 1, y
            new_box_x, new_box_y = x + 2, y
        elif action == 'L':
            new_x, new_y = x, y - 1
            box_x, box_y = x, y - 1
            new_box_x, new_box_y = x, y - 2
        elif action == 'R':
            new_x, new_y = x, y + 1
            box_x, box_y = x, y + 1
            new_box_x, new_box_y = x, y + 2
        else:
            raise Exception('Invalid action')

        if box_x is None:
            self_curr_replace_dict = {'Y': '_', 'T': 'X'}
            self_curr_replace_symbol = self_curr_replace_dict[curr_symbol]

            self_next_replace_dict = {'_': 'Y', 'X': 'T'}
            self_next_replace_symbol = self_next_replace_dict[n_state.board[new_x][new_y]]

            n_state.board[x][y] = self_curr_replace_symbol
            n_state.player_pos = new_x, new_y
            n_state.board[new_x][new_y] = self_next_replace_symbol
        else:
            self_curr_replace_dict = {'Y': '_', 'T': 'X'}
            self_curr_replace_symbol = self_curr_replace_dict[curr_symbol]

            self_next_replace_dict = {'O': 'Y', '@': 'T'}
            self_next_replace_symbol = self_next_replace_dict[n_state.board[new_x][new_y]]

            # box_curr_replace_dict = {'O': '_', '@': 'X'}
            # box_curr_replace_symbol = box_curr_replace_dict[n_state.board[box_x][box_y]]

            box_next_replace_dict = {'_': 'O', 'X': '@'}
            box_next_replace_symbol = box_next_replace_dict[n_state.board[new_box_x][new_box_y]]

            n_state.board[x][y] = self_curr_replace_symbol
            n_state.player_pos = new_x, new_y
            n_state.board[new_x][new_y] = self_next_replace_symbol
            n_state.board[new_box_x][new_box_y] = box_next_replace_symbol

        is_winner = n_state._count_free_boxes() == 0
        if n_state.terminal or is_winner:
            n_state.terminal = True
            n_state.win = is_winner

        return n_state

    ############################
    # State util functions
    ############################
    def _find_player(self):
        for x in range(len(self.board)):
            for y in range(len(self.board[x])):
                if self.board[x][y] in ['Y', 'T']:
                    return x, y
        return None

    def _count_free_boxes(self):
        return np.sum(self.board == 'O')

    def _count_boxes_on_goals(self):
        return np.sum(self.board == '@')

    def _serialize_state(self):
        str_board = ''.join(self.board.flatten())
        str_shape = ''.join([str(x) for x in self.board_size])
        s = str_shape + '|' + str_board
        return s

    def _deserialize_state(self, s):
        str_shape, str_board = s.split('|')
        board_size = [int(x) for x in str_shape]
        board = np.array([x for x in str_board]).reshape(board_size)
        return board

    def __str__(self):
        # pretty print the board
        ret = '\n'.join([' '.join(x) for x in self.board])
        ret += '\n'
        ret += 'History: ' + str(list(self.history.values())) + '\n'
        ret += 'Step count: {}\n'.format(self.step_count)
        ret += 'Free boxes: {}\n'.format(self._count_free_boxes())
        ret += 'Player pos: {}\n'.format(self.player_pos)
        ret += 'Terminal: {}\n'.format(self.terminal)
        ret += 'Win: {}\n'.format(self.win)
        ret += 'Actions: {}\n'.format(self.get_actions())
        return ret

    def render(self):
        return self.board.copy()


class BFS_Solver:
    def __init__(self):
        self.visited = set()
        self.queue = deque()

    def solve(self, state):
        self.queue.append(state)
        self.visited.add(state._serialize_state())
        start_time = datetime.now()
        while len(self.queue) > 0:
            curr_state = self.queue.popleft()
            msg_time_in_seconds = f'[{(datetime.now() - start_time).total_seconds()}]'
            msg_q = 'Queue size: {}'.format(len(self.queue))
            msg_v = 'Visited size: {}'.format(len(self.visited))
            msg_s = 'Step count: {}'.format(curr_state.step_count)
            msg_b = 'Free boxes: {}'.format(curr_state._count_free_boxes())
            msg_t = 'Terminal: {}'.format(curr_state.terminal)
            msg = '\t'.join([msg_time_in_seconds, msg_q, msg_v, msg_s, msg_b, msg_t])
            # print(curr_state)
            print(msg)
            # j = 3

            if curr_state.terminal:
                return curr_state
            for action in curr_state.get_actions():
                next_state = curr_state.apply_action(action)
                if next_state._serialize_state() not in self.visited:
                    self.queue.append(next_state)
                    self.visited.add(next_state._serialize_state())
        return None


if __name__ == '__main__':
    simple = [
        ['#', '#', '#', '#'],
        ['#', '_', 'Y', '#'],
        ['#', 'O', 'O', '#'],
        ['#', 'X', 'X', '#'],
        ['#', '#', '#', '#'],
    ]
    hellish = [
        ['#', '#', '#', '#', '#', '#', '#', '#'],
        ['#', '_', '_', '_', '#', '_', '_', '#'],
        ['#', '_', '#', '_', '#', 'O', 'X', '#'],
        ['#', '_', '_', '_', '_', 'O', 'X', '#'],
        ['#', '_', '#', '_', '#', 'O', 'X', '#'],
        ['#', '_', '_', '_', '#', '_', '_', '#'],
        ['#', '#', '#', '#', '#', 'Y', '_', '#'],
        ['#', '#', '#', '#', '#', '#', '#', '#'],
    ]
    extreme = [
        ['#', '#', '#', '#', '#', '#', '#'],
        ['#', '#', '#', '_', '_', '#', '#'],
        ['#', '#', '#', 'O', '_', '#', '#'],
        ['#', '_', 'O', 'Y', '_', '_', '#'],
        ['#', '_', 'X', 'X', 'X', '_', '#'],
        ['#', '#', 'O', '_', '_', '#', '#'],
        ['#', '#', '_', '_', '#', '#', '#'],
        ['#', '#', '#', '#', '#', '#', '#'],
    ]

    level = hellish

    sokoban = Sokoban()
    sokoban.start(level)

    n_state = sokoban
    sol = dict()

    agent = BFS_Solver()
    solution = agent.solve(n_state)
    print(solution)

    solution_actions = list(solution.history.values())
    s = ''
    i = 1
    for idx, action in enumerate(solution_actions):
        s += action
        if len(s) > 5 or idx == len(solution_actions) - 1:
            print(f'{i}: {s}')
            s = ''
            i += 1

    section_generate_image = True
    if section_generate_image:
        # Convert ASCII array to image
        color_map = {
            'WHITE': (255, 255, 255),
            'BLACK': (0, 0, 0),
            'RED': (255, 0, 0),
            'GREEN': (0, 255, 0),
            'BLUE': (0, 0, 255),
            'YELLOW': (255, 255, 0),
            'CYAN': (0, 255, 255),
            'MAGENTA': (255, 0, 255),
            'ORANGE': (255, 165, 0),
            'PURPLE': (128, 0, 128),
            'BROWN': (165, 42, 42),
            'PINK': (255, 192, 203),
            'GRAY': (128, 128, 128),
        }
        COLORS = {
            'Y': 'RED',  # color_map['RED'],
            'T': 'RED',  # color_map['RED'],

            'X': 'PINK',  # color_map['PINK'],
            '#': 'BLACK',  # lor_map['BLACK'],
            '_': 'WHITE',  # color_map['WHITE'],

            'O': 'YELLOW',  # color_map['YELLOW'],
            '@': 'ORANGE',  # color_map['ORANGE'],
        }

        all_imgs = []
        n_state = sokoban
        for action in solution_actions + ['X']:
            img = n_state.render()
            img_colors = [COLORS[img[i][j]] for i in range(len(img)) for j in range(len(img[0]))]
            img_colors = np.array(img_colors).reshape((len(img), len(img[0])))

            img_colors_code = [color_map[COLORS[img[i][j]]] for i in range(len(img)) for j in range(len(img[0]))]
            img_colors_code = np.array(img_colors_code).reshape((len(img), len(img[0]), 3)).astype(np.uint8)

            # Convert image to 28x28
            import cv2

            img_colors_code = cv2.resize(img_colors_code, (255, 255), interpolation=cv2.INTER_AREA)

            # Convert to PIL image and show
            all_imgs.append(img_colors_code)

            # print(img)
            # print("<-------->")
            # if action == 'X':
            #     break
            # n_state = n_state.apply_action(action)

            import imageio

            # Make gif from images according to image shape
            imageio.mimsave('sokoban.gif', all_imgs, duration=0.5,
                            subrectangles=True)  # subrectangles=True is important to avoid black borders
            print("GIF saved to disk.")
