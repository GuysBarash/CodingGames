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
    dprint("\npass")


def dprint_array(arr):
    for ridx in range(arr.shape[0]):
        dprint(''.join(arr[ridx]))


class World:
    def __init__(self):
        self.points_count = None
        self.points = None
        self.samples = None

    def visualize(self):
        import matplotlib.pyplot as plt
        plt.scatter(self.X, self.y)
        plt.show()
        j = 3

    def update(self, d=None):
        read_time = datetime.now()
        if d is None:
            self.points_count = int(input())
            self.points = [list()] * self.points_count
            for i in range(self.points_count):
                self.points[i] = [int(j) for j in input().split()]
            dprint("Update completed.")
        else:
            self.points_count = d['points_count']
            self.points = d['points']

        self.samples = np.array(self.points)
        self.X = self.samples[:, 0]
        self.y = self.samples[:, 1]
        read_time = (datetime.now() - read_time).total_seconds()
        dprint(f"Read time: {read_time}")

    def get_params(self):
        d = dict()
        d['points_count'] = self.points_count
        d['points'] = self.points
        return d


class Agent:
    def __init__(self, world):
        self.world = world

    def by_fit(self, X, y):
        from scipy.optimize import curve_fit

        funcs = dict()
        resdf = pd.DataFrame(columns=['title', 'perr', 'sqerr', 'coef', 'allcoef'])
        bound = 0.005

        # O(1)
        title = '1'
        funcs[title] = lambda x, p0: p0
        popt, pcov = curve_fit(funcs[title], X, y, p0=(1.))
        if popt[0] < bound:
            dprint(f"O({title}) was ruled out. Coef: {popt[0]}")
            popt[0] = 10
        perr = 0  # np.sqrt(np.diag(pcov))
        sqerr = np.sum(np.power(funcs[title](X, *popt) - y, 2)) * 0.5
        if sqerr < 0:
            sqerr = np.inf
        resdf.loc[title] = [title, perr, sqerr, popt[0], popt]

        # O(logn)
        title = 'log n'
        funcs[title] = lambda x, p1, p0: p1 * np.log(x) + p0
        popt, pcov = curve_fit(funcs[title], X, y, p0=(1., 1.))
        if popt[0] < bound:
            dprint(f"O({title}) was ruled out. Coef: {popt[0]}")
            popt[0] = 10
        perr = 0  # np.sqrt(np.diag(pcov))
        sqerr = np.sum(np.power(funcs[title](X, *popt) - y, 2))
        if sqerr < 0:
            sqerr = np.inf
        resdf.loc[title] = [title, perr, sqerr, popt[0], popt]

        # O(n)
        title = 'n'
        funcs[title] = lambda x, p1, p0: p1 * np.power(X, 1) + p0
        popt, pcov = curve_fit(funcs[title], X, y, p0=(1., 1.))
        if popt[0] < bound:
            dprint(f"O({title}) was ruled out. Coef: {popt[0]}")
            popt[0] = 10
        perr = 0  # np.sqrt(np.diag(pcov))
        sqerr = np.sum(np.power(funcs[title](X, *popt) - y, 2)) * 0.7
        if sqerr < 0:
            sqerr = np.inf
        resdf.loc[title] = [title, perr, sqerr, popt[0], popt]

        # O(nlogn)
        title = 'n log n'
        funcs[title] = lambda x, p2, p1, p0: (p1 * x) * (np.log(x + p2)) + p0
        try:
            popt, pcov = curve_fit(funcs[title], X, y, p0=(1., 1., 1.))
        except RuntimeError as e:
            popt = [1., 1., 1.]

        if popt[0] < bound or popt[1] < bound:
            dprint(f"O({title}) was ruled out. Coef: {popt[0],popt[1]}")
            popt[0] = 10
        perr = 0  # np.sqrt(np.diag(pcov))
        sqerr = np.sum(np.power(funcs[title](X, *popt) - y, 2))
        if sqerr < 0:
            sqerr = np.inf
        resdf.loc[title] = [title, perr, sqerr, popt[0], popt]

        # O(n^2)
        title = 'n^2'
        funcs[title] = lambda x, p2, p1, p0: p2 * np.power(x, 2) + p1 * np.power(x, 1) + p0
        popt, pcov = curve_fit(funcs[title], X, y, p0=(1., 1., 1.))
        if popt[0] < bound:
            dprint(f"O({title}) was ruled out. Coef: {popt[0]}")
            popt[0] = 10
        perr = 0  # np.sqrt(np.diag(pcov))
        sqerr = np.sum(np.power(funcs[title](X, *popt) - y, 2))
        if sqerr < 0:
            sqerr = np.inf
        resdf.loc[title] = [title, perr, sqerr, popt[0], popt]

        # O(n^2 log n)
        title = 'n^2 log n'
        funcs[title] = lambda x, p3, p2, p1, p0: p2 * np.power(x, 2) * (np.log(x + p3)) + p1 * np.power(x, 1)  # + p0
        try:
            init_cond = (1., 1., 1., 1.)
            popt, pcov = curve_fit(funcs[title], X, y, p0=init_cond)
        except RuntimeError as e:
            popt = init_cond

        if popt[0] < bound or popt[1] < bound:
            dprint(f"O({title}) was ruled out. Coef: {popt[0],popt[1]}")
            popt[0] = 10
        perr = 0  # np.sqrt(np.diag(pcov))
        sqerr = np.sum(np.power(funcs[title](X, *popt) - y, 2)) * 0.7
        if sqerr < 0:
            sqerr = np.inf
        resdf.loc[title] = [title, perr, sqerr, popt[0], popt]

        # O(n^3)
        title = 'n^3'
        funcs[title] = lambda x, p3, p2, p1, p0: p3 * np.power(X, 3) + p2 * np.power(X, 2) + p1 * np.power(X, 1) + p0
        popt, pcov = curve_fit(funcs[title], X, y, p0=(1., 1., 1., 1.))
        if popt[0] < bound:
            dprint(f"O({title}) was ruled out. Coef: {popt[0]}")
            popt[0] = 10
        perr = 0  # np.sqrt(np.diag(pcov))
        sqerr = np.sum(np.power(funcs[title](X, *popt) - y, 2))
        if sqerr < 0:
            sqerr = np.inf
        resdf.loc[title] = [title, perr, sqerr, popt[0], popt]

        resdf = resdf.sort_values(by='sqerr')

        # O(2^n)
        title = '2^n'
        if X.max() > 80:
            sqerr = np.inf
            perr = 0  # np.sqrt(np.diag(pcov))
            popt = [-1.]
            dprint(f"O({title}) was ruled out. X is too big to calculate.")
        else:
            funcs[title] = lambda x, p2, p1, p0: p2 * np.power(2 + p1, x) + p0
            popt, pcov = curve_fit(funcs[title], X, y, p0=(1., 1., 1.))
            if popt[0] < bound:
                dprint(f"O({title}) was ruled out. Coef: {popt[0]}")
                popt[0] = 10
            perr = 0  # np.sqrt(np.diag(pcov))
            sqerr = np.sum(np.power(funcs[title](X, *popt) - y, 2))
            if sqerr < 0:
                sqerr = np.inf

        resdf.loc[title] = [title, perr, sqerr, popt[0], popt]
        resdf = resdf.sort_values(by='sqerr')

        return resdf

    def observe(self):
        X = self.world.X
        y = self.world.y

        errs = dict()
        resdf = self.by_fit(X, y)

        print_dict(resdf['sqerr'])
        return resdf.index[0]


d = None
if LOCAL_MODE:
    d = dict()
    d["points_count"] = 90
    d["points"] = [[34, 89], [134, 760], [234, 1445], [334, 2933], [434, 5008], [534, 7692], [634, 10940], [734, 14274],
                   [834, 18840], [934, 23534], [1034, 28630], [1134, 34862], [1234, 40788], [1334, 47782],
                   [1434, 54809], [1534, 62810], [1634, 71086], [1734, 79948], [1834, 89340], [1934, 99465],
                   [2034, 109515], [2134, 120289], [2234, 131184], [2334, 144029], [2434, 156040], [2534, 169119],
                   [2634, 182308], [2734, 196256], [2834, 210583], [2934, 225883], [3034, 241551], [3134, 257257],
                   [3234, 274128], [3334, 291089], [3434, 308889], [3534, 327040], [3634, 346043], [3734, 365158],
                   [3834, 385165], [3934, 405247], [4034, 426228], [4134, 447276], [4234, 472780], [4334, 494214],
                   [4434, 519302], [4534, 538562], [4634, 561288], [4734, 585405], [4834, 611465], [4934, 636536],
                   [5034, 662347], [5134, 688659], [5234, 716543], [5334, 743685], [5434, 771559], [5534, 801174],
                   [5634, 829250], [5734, 860126], [5834, 889141], [5934, 919914], [6034, 950829], [6134, 983018],
                   [6234, 1014896], [6334, 1048154], [6434, 1081008], [6534, 1115426], [6634, 1149895], [6734, 1185505],
                   [6834, 1219942], [6934, 1256557], [7034, 1292808], [7134, 1338246], [7234, 1368160], [7334, 1414499],
                   [7434, 1449312], [7534, 1484492], [7634, 1523042], [7734, 1563109], [7834, 1603487], [7934, 1644975],
                   [8034, 1686382], [8134, 1729230], [8234, 1770767], [8334, 1815009], [8434, 1858475], [8534, 1903074],
                   [8634, 1947374], [8734, 1992645], [8834, 2038509], [8934, 2084824]]

world = World()
world.update(d)
if LOCAL_MODE:
    # world.visualize()
    pass
print_dict(world.get_params())

agent = Agent(world)
res = agent.observe()
time.sleep(0.01)
print(f'O({res})')
