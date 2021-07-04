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


class BFSagent:
    from collections import deque

    def __init__(self):
        self.root = None
        self.visited = dict()
        self.q = 1

    def observe(self, state):
        self.root = state
        visited, q = list(), deque([root.sig()])
