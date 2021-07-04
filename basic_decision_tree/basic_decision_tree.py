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
    dprint("")


class World:

    def __init__(self):
        self.data_points = None
        self.raw_data = None
        self.data = None

        self.root = None

    def update(self, d=None):
        if d is None:
            self.data_points = int(input())
            self.raw_data = list()
            for i in range(self.data_points):
                self.raw_data.append([int(j) for j in input().split(' ')])
        else:
            self.data_points = d['data_points']
            self.raw_data = list()
            for i in range(self.data_points):
                self.raw_data.append(d[f'raw_data_{i}'])

    def clean_data(self):
        self.data = pd.DataFrame(index=range(self.data_points),
                                 columns=['index', 'data', 'label'],
                                 data=self.raw_data
                                 )
        self.data = self.data.set_index('index')

    def get_params(self):
        d = dict()
        d['data_points'] = self.data_points
        for i in range(self.data_points):
            d[f'raw_data_{i}'] = self.raw_data[i]
        return d

    def get_data(self):
        return self.data['data']

    def get_labels(self):
        return self.data['label']


class DecisionTree:
    class Node:
        def __init__(self, data, labels, depth=0):
            self.data = data
            self.labels = labels
            self.depth = depth

            # Terminal node
            self.terminal = labels.unique().shape[0] <= 1 or data.unique().shape[0] <= 1
            self.label = None
            if self.terminal:
                unique_labels = labels.unique()
                if len(unique_labels) == 1:
                    self.label = unique_labels[0]
                else:
                    unique_data = data.unique()
                    if len(unique_data) == 1:
                        self.label = unique_labels[0]

            # Not terminal node
            self.seperator = None
            self.seperator_horn_size = None
            self.left_node = None
            self.right_node = None

        def find_seperator(self):
            # Find optimal seperator
            beetles = self.data.index.to_list()
            beetles_count = len(beetles)
            species_unique = self.labels.unique().tolist()
            seperators_df = pd.DataFrame(columns=['Seperator', 'Horn',
                                                  'group_1_entropy', 'group_1_count',
                                                  'group_2_entropy', 'group_2_count',
                                                  'weighted_entropy'
                                                  ],
                                         index=beetles,
                                         data=0,
                                         )
            for seperator in beetles:
                seperator_horn_size = self.data[seperator]
                seperators_df.loc[seperator, 'Seperator'] = seperator
                seperators_df.loc[seperator, 'Horn'] = seperator_horn_size

                groups = pd.DataFrame(columns=[1, 2], index=self.data.index)
                groups[1] = self.data < seperator_horn_size
                groups[2] = self.data >= seperator_horn_size

                for c_group in [1, 2]:
                    group_keys = groups[c_group]
                    species = self.labels[group_keys].unique().tolist()
                    sub_group_label = self.labels[group_keys]

                    n = len(species)
                    e_binary = 0
                    p_per_species = sub_group_label.value_counts()
                    p_per_species /= p_per_species.sum()
                    for c_species in species:
                        pi = p_per_species[c_species]
                        e_t = - pi * np.log2(pi)
                        e_binary += e_t
                    seperators_df.loc[seperator, f'group_{c_group}_entropy'] = e_binary
                    seperators_df.loc[seperator, f'group_{c_group}_count'] = group_keys.sum()

            for c_group in [1, 2]:
                seperators_df['weighted_entropy'] += (seperators_df[f'group_{c_group}_entropy'] * seperators_df[
                    f'group_{c_group}_count'] / beetles_count)

            optimal_idx = seperators_df['weighted_entropy'].idxmin()
            optimal_seperator = seperators_df.loc[optimal_idx, 'Seperator']
            horn_size = seperators_df.loc[optimal_idx, 'Horn']

            self.seperator = optimal_seperator
            self.seperator_horn_size = horn_size

        def split(self):
            if not self.terminal:
                self.find_seperator()

                # Calculate left side
                left_node_idxs = self.data < self.seperator_horn_size
                left_node_data = self.data[left_node_idxs]
                left_node_labels = self.labels[left_node_idxs]
                self.left_node = DecisionTree.Node(left_node_data, left_node_labels, self.depth + 1)
                self.left_node.split()

                right_node_idxs = ~ left_node_idxs
                right_node_data = self.data[right_node_idxs]
                right_node_labels = self.labels[right_node_idxs]
                self.right_node = DecisionTree.Node(right_node_data, right_node_labels, self.depth + 1)
                self.right_node.split()

    def __init__(self):
        self.root = None
        self.depth_map = dict()

    def train(self, data, labels):
        self.root = DecisionTree.Node(data, labels)
        self.root.split()

    def get_seperators_depth(self, node=None):
        if node is None:
            node = self.root

        if not node.terminal:
            self.depth_map[node.depth] = self.depth_map.get(node.depth, list()) + [node.seperator]
            self.get_seperators_depth(node.left_node)
            self.get_seperators_depth(node.right_node)


world = World()
d = None
if LOCAL_MODE:
    d = dict()
    d["data_points"] = 4
    d["raw_data_0"] = [0, 0, 1]
    d["raw_data_1"] = [1, 1, 1]
    d["raw_data_2"] = [2, 1, 2]
    d["raw_data_3"] = [3, 2, 2]

world.update(d)
print_dict(world.get_params())

world.clean_data()

clf = DecisionTree()
clf.train(world.get_data(), world.get_labels())
clf.get_seperators_depth()
print(f"{clf.depth_map[max(clf.depth_map.keys())][0]}")
