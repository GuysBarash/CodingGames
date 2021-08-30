import time
from datetime import datetime
import math
import random


def dprint(s=''):
    print(s, file=sys.stderr)


def randomPolicy(state, depth_cap=None):
    current_depth = 0
    if depth_cap is None:
        depth_cap = 99999999999999

    while (not state.is_terminal()) and (current_depth < depth_cap):
        try:

            action = random.choice(state.get_actions())
            state = state.apply_action(action)
            current_depth += 1
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))

    return state.get_reward()


class TreeNode:
    Nodes_explored = 0

    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.is_terminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}
        self.id = TreeNode.Nodes_explored
        TreeNode.Nodes_explored += 1

    def __str__(self):
        s = []
        s.append("totalReward: %s" % (self.totalReward))
        s.append("numVisits: %d" % (self.numVisits))
        s.append("isTerminal: %s" % (self.isTerminal))
        s.append("possibleActions: %s" % (self.children.keys()))
        return "%s: {%s}" % (self.__class__.__name__, ', '.join(s))


class Mcts:
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2),
                 depth_cap=None,
                 rolloutPolicy=randomPolicy):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy
        self.depth_cap = depth_cap

    def search(self, initialState, needDetails=False):
        self.root = TreeNode(initialState, None)

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            round = 1
            while time.time() < timeLimit:
                round += 1
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()

        bestChild = self.getBestChild(self.root, 0)
        action = (action for action, node in self.root.children.items() if node is bestChild).__next__()
        if needDetails:
            return {"action": action, "expectedReward": bestChild.totalReward / bestChild.numVisits}
        else:
            return action

    def executeRound(self):
        """
            execute a selection-expansion-simulation-backpropagation round
        """
        node = self.selectNode(self.root)
        reward = self.rollout(node.state, depth_cap=self.depth_cap)
        self.backpropogate(node, reward)

    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        actions = node.state.get_actions()
        for action in actions:
            if action not in node.children:
                newNode = TreeNode(node.state.apply_action(action), node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode

        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = node.state.get_current_player() * child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)


class Agent:
    def __init__(self, raw=None):
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

        # self.engine = Mcts(iterationLimit=500, depth_cap=10)
        self.engine = MinMax(depth_cap=5, time_cap=None)

    def get_action(self, state):
        action = self.engine.search(state)
        dprint(f'Nodes: {self.engine.nodes_developed}')
        dprint(f'Time: {datetime.now() - self.engine.current_start_time}')
        dprint(f'Time violation: {self.engine.time_cap_violation}')
        return action
