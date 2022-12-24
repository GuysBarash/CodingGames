import sys
import socket
import re
import time
from collections import deque
from copy import deepcopy as copy
from queue import PriorityQueue

DEBUG_MODE = True
LOCAL_MODE = socket.gethostname() == 'Barash-pc'


def dprint(s=''):
    print(s, file=sys.stderr, flush=True)


def dprint_map(m):
    dprint("")
    for mt in m:
        dprint(mt)
    dprint("")


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
    def __init__(self, d_start=None, turn=0):
        self.d = d_start

        self.lines = None
        self.map = None
        self.robots = None
        self.robot_count = None
        self.turn = turn
        self.terminal = False
        self.markers = list()
        self.markers_count = 0

    def export(self):
        # Convert all to dict
        d = dict()
        d['lines'] = self.lines
        d['robots'] = self.robots
        d['robot_count'] = self.robot_count
        d['turn'] = self.turn
        d['terminal'] = self.terminal
        d['score'] = self.score
        d['markers'] = self.markers
        d['markers_count'] = self.markers_count
        self.d = d

        return copy(d)

    def export_and_print(self, dict_mode=False):
        if dict_mode:
            d = self.export()
            print_dict(d)
        else:
            d = self.export()
            dx = {k: v for k, v in d.items() if k not in ['robots']}
            robots = d['robots']
            for r in robots:
                msg = ''
                rx = {k: v for k, v in r.items() if k not in ['hist']}
                msg += f"{rx}\n"
                msg += f"{r['hist']}"
                dprint(msg)
            print_dict(dx)
        dprint_map(self.map)

    def place_robot(self):
        for r in self.robots:
            if r['terminated']:
                continue
            x, y = r['pos']
            direction = r['dir']
            self.map[y] = self.map[y][:x] + direction + self.map[y][x + 1:]

    def place_marker(self):
        for x, y, direction in self.markers.values():
            dir_sign = direction.lower()
            self.map[y] = self.map[y][:x] + dir_sign + self.map[y][x + 1:]

    def update(self, d=None):
        if d is None:
            # Get turn parameters from CLI
            self.lines = list()
            for i in range(10):
                line = input().lower()
                self.lines += [line]

            self.robot_count = int(input())
            self.robots = list()
            for i in range(self.robot_count):
                inputs = input().split()
                x = int(inputs[0])
                y = int(inputs[1])
                direction = inputs[2]

                r = dict()
                r['pos'] = x, y
                r['dir'] = direction
                r['idx'] = i
                r['terminated'] = False
                r['hist'] = [f'{x}.{y}.{direction}']
                self.robots += [r]

            self.terminal = False
            self.score = 0

            # Get position of markers
            self.markers = dict()
            self.marker_count = 0
            for y, line in enumerate(self.lines):
                for x, c in enumerate(line):
                    if c in ['u', 'd', 'r', 'l']:
                        self.markers[f'{x}x{y}'] = (x, y, c.upper())
                        self.marker_count += 1
            self.lines = [re.sub(r'[\.udrl]', '0', lt) for lt in self.lines]
        else:
            # Get turn parameters from d
            self.d = d
            self.lines = d['lines']
            self.robots = d['robots']
            self.robot_count = d['robot_count']
            self.turn = d['turn']
            self.terminal = d['terminal']
            self.markers = d['markers']
            self.markers_count = d['markers_count']
            self.score = d['score']

        self.max_y, self.max_x = len(self.lines), len(self.lines[0])

        self.map = copy(self.lines)
        self.place_marker()
        self.place_robot()
        return None

    def apply(self, action):
        n_state = State()
        d = self.export()

        action_x_str, action_y_str, action_dir = action.split(' ')
        action_x, action_y = int(action_x_str), int(action_y_str)
        marker_sig = f'{action_x}x{action_y}'
        d['markers'][marker_sig] = (action_x, action_y, action_dir)
        d['markers_count'] += 1

        n_state.update(d)
        return n_state

    def step(self):
        n_state = State()
        d = self.export()
        d['turn'] += 1
        for robot in d['robots']:
            if robot['terminated']:
                continue

            x, y = robot['pos']
            marker_sig = f'{x}x{y}'
            marker = self.markers.get(marker_sig, None)
            if marker is not None:
                robot['dir'] = marker[2]

            direction = robot['dir']
            hist_item = f'{x}.{y}.{direction}'

            old_x, old_y = x, y
            if direction == 'U':
                y -= 1
            elif direction == 'D':
                y += 1
            elif direction == 'R':
                x += 1
            elif direction == 'L':
                x -= 1

            y = y % self.max_y
            x = x % self.max_x

            new_sig = f'{x}.{y}.{direction}'
            if new_sig in robot['hist']:
                robot['terminated'] = True
                d['robot_count'] -= 1
                # dprint(f'Robot {robot["idx"]} in a loop')
                continue
            else:
                robot['hist'] += [new_sig]

            marker = self.map[y][x]
            if marker == '#':
                robot['terminated'] = True
                d['robot_count'] -= 1
                continue

            robot['pos'] = x, y

        if d['robot_count'] <= 0:
            d['terminal'] = True

        d['score'] += d['robot_count']
        d['lines'][old_y] = d['lines'][old_y][:old_x] + '1' + d['lines'][old_y][old_x + 1:]

        n_state.update(d)
        return n_state

    def get_actions_as_string(self):
        return ' '.join([f'{x} {y} {dir}' for (x, y, dir) in self.markers.values()])

    def get_actions(self):
        actions = list()
        actions += ['@ @ @']
        for r in self.robots:
            if r['terminated']:
                continue
            x, y = r['pos']
            direction = r['dir']
            curr_marker = self.lines[y][x]
            if curr_marker == '0':
                for dir_t in ['U', 'D', 'R', 'L']:

                    if dir_t == 'U':
                        if self.lines[(y - 1) % self.max_y][x] != '#':
                            actions += [f'{x} {y} {dir_t}']
                    elif dir_t == 'D':
                        if self.lines[(y + 1) % self.max_y][x] != '#':
                            actions += [f'{x} {y} {dir_t}']
                    elif dir_t == 'R':
                        if self.lines[y][(x + 1) % self.max_x] != '#':
                            actions += [f'{x} {y} {dir_t}']
                    elif dir_t == 'L':
                        if self.lines[y][(x - 1) % self.max_x] != '#':
                            actions += [f'{x} {y} {dir_t}']
                    else:
                        pass
        return actions

    def get_sig(self):
        sig_robots = '@'.join([str(r) for r in self.robots])
        sig_markers = '@'.join([str(self.markers[k]) for k in sorted(self.markers.keys())])
        # Get self.markers values as a list ordered by their keys
        return sig_robots + '$' + sig_markers

    def __gt__(self, other):
        return self.score > other.score

    def __lt__(self, other):
        return self.score < other.score


class Agent:
    def __init__(self, state=None):
        self.start_time = time.time()

    def observe_bfs(self, state=None):
        # Create an empty queue
        q = deque()
        curr_state = copy(state)
        curr_state_score = curr_state.score
        best_score = curr_state_score

        q.append(curr_state)
        moves = list()
        round = -1
        visited = dict()
        terminals = dict()
        while len(q) > 0:
            dprint(f"Explored states: {len(visited)}\tBest: {best_score}\tQueue: {len(q)}")
            curr_state = q.popleft()
            visited[curr_state.get_sig()] = True
            round += 1
            actions = curr_state.get_actions()
            for action in actions:
                if action == '@ @ @':
                    n_state = curr_state.step()
                else:
                    n_state = curr_state.apply(action)
                if n_state.get_sig() not in visited:
                    best_score = max(best_score, n_state.score)
                    if n_state.terminal:
                        terminals[n_state.get_sig()] = (n_state, n_state.score)
                    else:
                        q.append(n_state)

        winner_state = max(terminals.values(), key=lambda x: x[1])[0]
        return winner_state.get_actions_as_string()

    def observe_astar(self, state=None):
        # Create an empty queue
        q = PriorityQueue()
        curr_state = copy(state)
        curr_state_score = curr_state.score
        best_score = curr_state_score

        q.put((self._get_state_score(curr_state), curr_state))
        round = -1
        visited = dict()
        calculated = dict()
        while not q.empty():
            duration = time.time() - self.start_time
            if duration > 0.85:
                dprint(f"Time out. Explored states: {len(visited)}\tBest: {best_score}\tQueue: {q.qsize()}")
                break

            curr_rank, curr_state = q.get()
            visited[curr_state.get_sig()] = True
            round += 1
            actions = curr_state.get_actions()

            if round % 1000 == 0:
                msg = ''
                msg += f'[{round:>5d}]'
                msg += f'[Score {-curr_rank:>5d}/{best_score:>5d}]'
                msg += f'[Queue {q.qsize():>5d}]'
                msg += f'[Visited {len(visited):>5d}]'
                msg += f'[Time {duration:>5.2f}]'
                dprint(msg)
            for action in actions:
                if action == '@ @ @':
                    n_state = curr_state.step()
                else:
                    n_state = curr_state.apply(action)
                if n_state.get_sig() not in visited:
                    best_score = max(best_score, n_state.score)
                    calculated[n_state.get_sig()] = (n_state, n_state.score)
                    if n_state.terminal:
                        pass
                    else:
                        q.put((-self._get_state_score(n_state), n_state))

        winner_state = max(calculated.values(), key=lambda x: x[1])[0]
        return winner_state.get_actions_as_string()

    def run_action(self, state, action=None, display=False):
        n_state = copy(state)
        if action is not None:
            n_state.apply(action)
        while True:
            n_state = n_state.step()
            if display:
                n_state.export_and_print()
                # time.sleep(0.3)
            if n_state.terminal:
                if display:
                    n_state.export_and_print()
                break

        return n_state

    def get_possible_actions(self, state):
        markers = list()
        hists = [r['hist'] for r in state.robots]
        flat_hists = [item for sublist in hists for item in sublist]
        hist_coords = [tuple(map(int, h.split('.')[0:2])) for h in flat_hists]
        hist_coords = list(set(hist_coords))

        # Remove occupied coords
        occupied_coords = [(x, y) for x, y, _ in state.markers]
        hist_coords = [c for c in hist_coords if c not in occupied_coords]

        dirs = ['U', 'D', 'R', 'L']

        possibles = list()
        for dirc in dirs:
            possibles += [(x, y, dirc) for (x, y) in hist_coords]

        return possibles

    def _get_state_score(self, state):
        reg_score = state.score
        robot_score = state.robot_count
        marker_score = state.markers_count
        score = (1 * reg_score) + (100 * robot_score) + (1000 * marker_score)
        return score


d = None
if LOCAL_MODE:
    d = None

state = State(d)
agent = Agent()

section_main = True
if section_main:
    if LOCAL_MODE:
        d = dict()
        d["lines"] = ['#00000000000000000#', '#00000000000000000#', '#00000000000000000#', '#00000000000000000#',
                      '#00000000#00000000#', '#00000000000000000#', '#00000000000000000#', '#00000000000000000#',
                      '#00000000000000000#', '###################']
        d["robots"] = [{'pos': (1, 0), 'dir': 'R', 'idx': 0, 'terminated': False, 'hist': ['1.0.R']},
                       {'pos': (17, 0), 'dir': 'D', 'idx': 1, 'terminated': False, 'hist': ['17.0.D']},
                       {'pos': (1, 8), 'dir': 'U', 'idx': 2, 'terminated': False, 'hist': ['1.8.U']},
                       {'pos': (17, 8), 'dir': 'L', 'idx': 3, 'terminated': False, 'hist': ['17.8.L']}]
        d["robot_count"] = 4
        d["turn"] = 0
        d["terminal"] = False
        d["score"] = 0
        d["markers"] = {'9x0': (9, 0, 'L'), '9x1': (9, 1, 'R'), '9x2': (9, 2, 'L'), '9x3': (9, 3, 'R'),
                        '1x4': (1, 4, 'D'), '2x4': (2, 4, 'U'), '3x4': (3, 4, 'D'), '4x4': (4, 4, 'U'),
                        '5x4': (5, 4, 'D'), '6x4': (6, 4, 'U'), '7x4': (7, 4, 'D'), '8x4': (8, 4, 'U'),
                        '10x4': (10, 4, 'U'), '11x4': (11, 4, 'D'), '12x4': (12, 4, 'U'), '13x4': (13, 4, 'D'),
                        '14x4': (14, 4, 'U'), '15x4': (15, 4, 'D'), '16x4': (16, 4, 'U'), '17x4': (17, 4, 'D'),
                        '9x5': (9, 5, 'R'), '9x6': (9, 6, 'L'), '9x7': (9, 7, 'R'), '9x8': (9, 8, 'L')}
        d["markers_count"] = 0


    else:
        pass

    state.update(d)
    state.export_and_print(dict_mode=True)
    action = agent.observe_astar(state)
    print(action)

    if LOCAL_MODE:
        action_list = [t for t in re.findall(r'(\d+ \d+ [UDLR])', action)]
        for action_t in action_list:
            # dprint("Action: {}".format(action_t))
            state = state.apply(action_t)
            state.export_and_print()
        while True:
            state = state.step()
            state.export_and_print()
            if state.terminal:
                # state.export_and_print()
                time.sleep(0.1)
                print(action)
                break
