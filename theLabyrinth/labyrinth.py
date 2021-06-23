import sys
import math
import collections
from copy import copy
import pandas as pd


def dprint(s):
    print(s, file=sys.stderr)


def rplce(s, position, c):
    st = s[:position] + c + s[position + 1:]
    return st


class Kirk:
    def _get_target(self):
        targets_per_mode = dict()
        targets_per_mode[0] = '?'
        targets_per_mode[1] = 'C'
        targets_per_mode[2] = 'T'
        targets_per_mode[3] = 'T'
        return targets_per_mode.get(self.mode, 'ERROR')

    def _find_item(self, target):
        row = self.maze[self.maze.eq(target).any(1)]
        if len(row) == 0:
            return None
        else:
            row = row.iloc[0]
            ridx = row.name
            cidx = row[row == target].index[0]
            return ridx, cidx

    def _mode_to_string(self):
        targets_per_mode = dict()
        targets_per_mode[0] = 'Exploring'
        targets_per_mode[1] = 'Seeking alternating rounte'
        targets_per_mode[2] = 'Going to C'
        targets_per_mode[3] = 'Going to T'
        return targets_per_mode.get(self.mode, 'ERROR')

    def __init__(self, r, c, a):
        self.alarm_count = a
        self.turn = -1
        self.maze_size = (r, c)
        self.r = r
        self.c = c

        self.pos_row = -1
        self.pos_col = -1
        self.last_target = None

        self.mode = 0
        # mode 0: C is unreachable\unknown
        # mode 1: C to T is too long
        # mode 2: Going to C
        # mode 3: Going to T
        self.current_path = None

        self.maze = pd.DataFrame(columns=range(c),
                                 index=range(0, r + 0),
                                 data='?'
                                 )

    def update(self, debug=None):
        self.turn += 1
        if debug is None:
            self.pos_row, self.pos_col = [int(i) for i in input().split()]
            for i in range(self.r):
                row = list(input())  # C of the characters in '#.TC?' (i.e. one line of the ASCII maze).
                self.maze.iloc[i] = row
        else:
            self.mode = debug[0]
            self.pos_row, self.pos_col = debug[1], debug[2]
            for i in range(self.r):
                row = list(debug[i + 3])  # C of the characters in '#.TC?' (i.e. one line of the ASCII maze).
                self.maze.iloc[i] = row

        current_item = self.maze.loc[self.pos_row, self.pos_col]
        dprint(f'Current item: {current_item}\t Target: {self._get_target()}')
        if self.mode != 3 and current_item == 'C':
            self.mode = 3
            self.current_path = None
            dprint(f'Change mode!')

    def show(self, player=None):
        dprint(f'MODE: {self._mode_to_string()}[{self.mode}]\tTurn: {self.turn}/{self.alarm_count}')
        dprint(f'Player: {self.pos_row}x{self.pos_col}')
        for ridx in self.maze.index:
            r = ''.join(self.maze.loc[ridx])
            if player is not None and ridx == self.pos_row:
                r = rplce(r, self.pos_col, '@')
            dprint(r)

    def act(self):
        if self.mode == 0:
            c_pos = self._find_item('C')
            if c_pos is None:
                step = self.explore()
                return step
            elif c_pos == (self.pos_row, self.pos_col):
                self.mode = 3
            else:
                t_pos = self._find_item('T')
                self.current_path = None
                step = self.findpath('T', c_pos)
                if step is None:
                    step = self.explore()
                    return step
                path_len = len(self.current_path)
                self.current_path = None
                if path_len <= self.alarm_count:
                    # Short enough
                    self.mode = 2
                else:
                    # Got a path, but it is too long
                    self.mode = 1

        if self.mode == 1:
            step = self.explore()
            self.mode = 0
            return step

        if self.mode == 2:
            step = self.findpath('C')

        if self.mode == 3:
            step = self.findpath('T')

        return step

    def explore(self):
        return self.findpath(target='?')

    def findpath(self, target=None, src=None):
        if self.current_path is None:
            # dprint(f"Seeking target: {target}")

            class Node:
                def __init__(self, r, c, last_move=None, prev_node=None, item='.'):
                    self.r = r
                    self.c = c
                    self.path = list()
                    self.sig = f'{self.r}_{self.c}'
                    self.item = item
                    if prev_node is not None:
                        self.path = copy(prev_node.path)
                    if last_move is not None:
                        self.path += [last_move]

                def get(self):
                    return f'{self.r}_{self.c}'

            # BFS
            nodes = dict()
            r, c = self.pos_row, self.pos_col
            if src is not None:
                r, c = src
            root = Node(r, c, item=self.maze.loc[r, c])
            nodes[root.sig] = root
            visited, q = list(), collections.deque([root.sig])

            while len(q) > 0 and self.current_path is None:
                node_sig = q.popleft()
                node = nodes.get(node_sig)
                visited.append(node_sig)

                for dir in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                    r = node.r
                    c = node.c
                    if dir == 'UP':
                        r -= 1
                    elif dir == 'DOWN':
                        r += 1
                    elif dir == 'LEFT':
                        c -= 1
                    elif dir == 'RIGHT':
                        c += 1
                    else:
                        raise Exception("BAD DIR")

                    if c <= 0 or c >= self.maze_size[1]:
                        continue
                    if r <= 0 or r >= self.maze_size[0]:
                        continue

                    item = self.maze.loc[r, c]
                    new_node = Node(r, c, dir, node, item)
                    if new_node.item == target:
                        # HIT
                        self.current_path = new_node.path
                        if target == '?' and len(self.current_path) > 1:
                            # the "?" might hide a wall. so we step but do not touch
                            self.current_path = self.current_path[:-1]
                        break
                    elif new_node.item in ['#', '?', 'C']:
                        continue
                    elif new_node.sig in visited:
                        continue

                    else:
                        # Keep exploring
                        q.append(new_node.sig)
                        nodes[new_node.sig] = new_node
                        visited.append(new_node.sig)

        if self.current_path is None:
            return None
        else:
            dprint(f"Path [{len(self.current_path)}]: {self.current_path}")
            chosen_step = self.current_path[0]
            self.current_path = self.current_path[1:]
            if len(self.current_path) == 0:
                self.current_path = None
            return chosen_step


# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

# r: number of rows.
# c: number of columns.
# a: number of rounds between the time the alarm countdown is activated and the time the alarm goes off.
r, c, a = 15, 30, 7
# r, c, a = [int(i) for i in input().split()]
dprint(f"Initial params: {r},{c},{a}")
kirk = Kirk(r, c, a)
dprint("done")

# game loop
while True:
    # kr: row where Kirk is located.
    # kc: column where Kirk is located.
    # kr, kc = [int(i) for i in input().split()]
    # for i in range(r):
    #     row = input()  # C of the characters in '#.TC?' (i.e. one line of the ASCII maze).
    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr)
    # Kirk's next move (UP DOWN LEFT or RIGHT).
    # print("RIGHT")

    debug_line = list()
    debug_line += [0]
    debug_line += [11, 3]
    r = '''?#####?????????????###########
?#.###????####.????###########
#...#.######.#.????.#......T##
###.#.######.#.######.########
##...........#.######.########
###.#.######......###.##??????
#...#.###....#.##.###.##??????
#.####????####.##.....##??????
#.....?????????#########??????
###.##?????????####.....??????
###.##????????????????????????
?##@..????????????????????????
?#####????????????????????????
?###C.????????????????????????
??????????????????????????????'''
    debug_line += r.split('\n')

    kirk.update(debug_line)
    kirk.show('@')
    step = kirk.act()
    print(step)
