import sys
import math
import numpy as np

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


class Ground:
    def __init__(self):
        self.gravity = 3.711
        self.l = None
        self.n = None

        self.landing_x = None
        self.landing_y = None

    def update(self, inp=None):
        # land_x: X coordinate of a surface point. (0 to 6999)
        # land_y: Y coordinate of a surface point. By linking all the points together in a sequential fashion, you form the surface of Mars.

        if inp is None:
            surface_n = int(input())  # the number of points used to draw the surface of Mars.
            l = list()
            for i in range(surface_n):
                land_x, land_y = [int(j) for j in input().split()]
                l += [(land_x, land_y)]
            msg = 'Map input:\t[' + ', '.join([f"({x}, {y})" for x, y in l]) + ']' + '\n'
            dprint(msg)
        else:
            l = inp

        self.gravity = 3.711
        self.l = l
        self.n = len(l)
        for i in range(1, self.n):
            prev = l[i - 1]
            curr = l[i]
            if curr[1] == prev[1]:
                self.landing_x = (curr[0], prev[0])
                self.landing_y = curr[1]

    def get_landing_strip(self):
        return f'{self.landing_x[0]}x{self.landing_y} <--> {self.landing_x[1]}x{self.landing_y}'

    def get_strip_size(self):
        return int(np.abs(self.landing_x[0] - self.landing_x[1]))

    def get_landing(self, symbol='XY'):
        landing_x = sum(self.landing_x) / len(self.landing_x)
        landing_y = self.landing_y
        if symbol in ['X', 'x']:
            return landing_x
        elif symbol in ['Y', 'y']:
            return landing_y
        else:
            return landing_x, landing_y

    def get_map(self):
        msg = ''
        msg += 'Map:' + '\n'
        for i in range(self.n):
            x, y = self.l[i]
            msg += f'[{i:>3}]\t{x:>3}x{y:>3}'
            if x in self.landing_x:
                msg += '  <----'
            msg += '\n'
        return msg


class Craft:
    def __init__(self, ground):
        self.turn = 0
        self.ground = ground
        self.x = 0
        self.y = 0

        self.x_speed = 0
        self.y_speed = 0

        self.x_acc = 0
        self.y_acc = 0

        self.fuel = 9999999
        self.rotate = 0
        self.power = 0

        self.inp = None
        self.landing_speed_x = 20
        self.landing_speed_y = 40
        self.gravity = ground.gravity

    def update(self, d=None):
        # h_speed: the horizontal speed (in m/s), can be negative.
        # v_speed: the vertical speed (in m/s), can be negative.
        # fuel: the quantity of remaining fuel in liters.
        # rotate: the rotation angle in degrees (-90 to 90).
        # power: the thrust power (0 to 4).

        self.turn += 1
        if d is None:
            self.x, self.y, x_speed, y_speed, self.fuel, self.rotate, self.power = [int(i) for i in input().split()]
            self.x_acc = x_speed - self.x_speed
            self.y_acc = y_speed - self.y_speed
            self.x_speed = x_speed
            self.y_speed = y_speed
        else:
            self.x = d["x"]
            self.y = d["y"]
            self.x_speed = d["x_speed"]
            self.y_speed = d["y_speed"]
            self.fuel = d["fuel"]
            self.rotate = d["rotate"]
            self.power = d["power"]
            self.x_acc = d['x_acc']
            self.y_acc = d['y_acc']

        gx, gy = self.ground.get_landing()
        self.distance_x = gx - self.x
        self.distance_y = self.y - gy

    def act(self):
        single_x_unit = self.ground.get_strip_size() / 2
        x_units_from_target = np.abs(self.distance_x) / single_x_unit
        x_units_from_target = min(4, x_units_from_target) / 4.0
        ntilt = 45 / x_units_from_target
        ntilt_dir = np.sign(- self.distance_x)
        tilt = int(ntilt_dir * ntilt)

        npower = self.calc_y_speed()
        return f"{tilt} {npower}"

    def get_params(self):
        d = dict()
        d['x'], d['y'] = self.x, self.y
        d['x_speed'], d['y_speed'] = self.x_speed, self.y_speed
        d['x_acc'], d['y_acc'] = self.x_acc, self.y_acc
        d['fuel'], d['rotate'], d['power'] = self.fuel, self.rotate, self.power
        d['distance_x'] = self.distance_x
        d['distance_y'] = self.distance_y
        return d

    def get_statues(self):
        msg = ''
        msg += 'Craft:' + '\n'
        msg += f"[{self.turn}]" + '\n'
        msg += f'Last input: {self.inp}' + '\n'
        msg += f"{self.x}\t{self.y}" + '\n'
        msg += f"{self.x_speed}\t{self.y_speed}" + '\n'
        msg += f"{self.x_acc}\t{self.y_acc}" + '\n'
        msg += f"Fuel: {self.fuel}\tRotate: {self.rotate}\tPower: {self.power}" + '\n'

        return msg

    def calc_y_speed(self):

        max_y = 2600
        min_y = 0

        max_speed = -40
        min_speed = -10

        std_unit = 0.1
        min_std_unit = 2

        relative_y = (self.distance_y - min_y) / (max_y - min_y)
        expected_relative_speed = (max_speed - min_speed) * relative_y
        optimal_speed = min_speed + expected_relative_speed
        speed_std = min(optimal_speed * std_unit, min_std_unit)

        speed_delta = (self.y_speed - optimal_speed) / speed_std
        dprint(f"Optimal speed: {optimal_speed:>.1f}\t Current: {self.y_speed:>.1f}")
        dprint(f"Speed difference from optimal: {speed_delta:>.1f}")
        desired_thrust = min(4, max(1, int(np.floor(speed_delta))))
        return desired_thrust


d = None
if LOCAL_MODE:
    d = [(0, 100), (1000, 500), (1500, 1500), (3000, 1000), (4000, 150), (5500, 150), (6999, 800)]

ground = Ground()
ground.update(d)
craft = Craft(ground)
dprint(ground.get_map())

# game loop
while True:
    d = None
    if LOCAL_MODE:
        d = dict()
        d["x"] = 2789
        d["y"] = 1552
        d["x_speed"] = 23
        d["y_speed"] = -87
        d["x_acc"] = 1
        d["y_acc"] = -4
        d["fuel"] = 523
        d["rotate"] = -68
        d["power"] = 1

    craft.update(d)
    print_dict(craft.get_params())
    command = craft.act()
    dprint(f"[{craft.turn}] Command:\t{command}")
    print(command)

    if LOCAL_MODE:
        print("LOCAL MODE. BREAK.")
        break
