import sys
import math


def dprint(s=''):
    print(s, file=sys.stderr)


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

    def update(self, inp=None):
        # h_speed: the horizontal speed (in m/s), can be negative.
        # v_speed: the vertical speed (in m/s), can be negative.
        # fuel: the quantity of remaining fuel in liters.
        # rotate: the rotation angle in degrees (-90 to 90).
        # power: the thrust power (0 to 4).

        self.turn += 1
        if inp is None:
            self.inp = [int(i) for i in input().split()]
        else:
            self.inp = inp
        self.x, self.y, x_speed, y_speed, self.fuel, self.rotate, self.power = self.inp
        self.x_acc = x_speed - self.x_speed
        self.y_acc = y_speed - self.y_speed
        self.x_speed = x_speed
        self.y_speed = y_speed

    def get_params(self):
        d = dict()
        d['x'], d['y'] = self.x, self.y
        d['x_speed'], d['y_speed'] = self.x_speed, self.y_speed
        d['x_acc'], d['y_acc'] = self.x_acc, self.y_acc
        d['fuel'], d['rotate'], d['power'] = self.fuel, self.rotate, self.power
        return d

    def get_target(self):
        d = dict()
        d['x'], d['y'] = ground.get_landing()
        d['x_speed'], d['y_speed'] = int(self.landing_speed_x / 2.0), int(self.landing_speed_y / 2.0)
        d['rotate'] = 0
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

    def build_course(self):
        curr = self.get_params()
        trgt = self.get_target()

        x_dist = curr['x'] - trgt['x']
        y_dist = curr['y'] - trgt['y']

        # TODO
        return "90 00"


ground = Ground()
ground.update([(0, 100), (1000, 500), (1500, 100), (3000, 100), (5000, 1500), (6999, 1000)])
craft = Craft(ground)
dprint(ground.get_map())

# game loop
while True:
    craft.update([2500, 2500, 0, 0, 500, 0, 0])
    dprint(craft.get_statues())
    command = craft.build_course()
    dprint(f"[{craft.turn}] Command:\t{command}")
    print(command)

    # rotate power. rotate is the desired rotation angle. power is the desired thrust power.
