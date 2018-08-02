import datetime
import numpy as np
import math
from src.general_player import Player
from src.config import MINIMUM_TEMPERATURE_ACCEPTED, DEFAULT_TRAIN_THINK_TIME, DEFAULT_NUMBER_OF_SIMULATIONS

from importlib import reload
import src.tree_node
reload(src.tree_node)

from src.tree_node import Node

class MCTS():
    def __init__(self, model, **kwargs):
        self.seconds = kwargs.get('time', DEFAULT_TRAIN_THINK_TIME)
        self.limit = kwargs.get('limit', DEFAULT_NUMBER_OF_SIMULATIONS)
        self.learning = kwargs.get('learning', False)
        self.model = model
        self.time_limit = datetime.timedelta(seconds=self.seconds)

    def search(self, **kwargs):
        limit = kwargs.get('limit', self.limit)
        root = kwargs.get('root', None)
        state = kwargs.get('state', None)
        if not root is None:
            root.parent = None
            return self.search_node(root, simulation_limit=limit)
        if not state is None:
            root = Node(state)
            return self.search_node(root, simulation_limit=limit)
        raise Exception('Invalid MCTS search arguments')

    def search_node(self, root_node, simulation_limit=None):
        self.root_node = root_node
        if not root_node.is_visited():
            root_node.evaluate(self.model)
            self.__add_dirichlet_noise()
            root_node.N = 1
        if simulation_limit is None:
            begin = datetime.datetime.utcnow()
            while datetime.datetime.utcnow() - begin < self.time_limit: #TODO tread this
                self.run_one_simulation()
            self.search_time = self.seconds
        else:
            begin = datetime.datetime.utcnow()
            for i in range(simulation_limit):
                self.run_one_simulation()
            self.search_time = (datetime.datetime.utcnow() - begin).total_seconds()
        return self.root_node

    def __add_dirichlet_noise(self):
        n = self.root_node.state.action_space_size()
        self.root_node.children_p = (0.75 * self.root_node.children_p +
                                     0.25 * np.random.dirichlet([1./n] * n))

    def run_one_simulation(self):
        """Traverse to a leaf and update statistics with back prop"""
        last_node = self.simulate_to_leaf()
        self.backpropagation(last_node)

    def simulate_to_leaf(self):
        """Iteratively select a child from the current node until a leaf or
         a terminal node is encountered (which is inherently a leaf)"""

        node = self.root_node
        while not node.is_terminal:
            move_index = self.select(node)
            new_node = node.children[move_index]
            if new_node is None:
                node = node.expand_and_evaluate(move_index, self.model)
                break
            node = new_node
        return node

    def select(self, node):
        "Select a move from node and return it"
        move_indexes = range(len(node.children))
        scores = np.array([self.UCT_score(node, move) for move in move_indexes])
        best_index = np.argmax(scores)
        return best_index

    def UCT_score(self, parent, index):
        """Calculate the uppper confidence bound score as it was calculated in the AlphaZero paper"""
        node = parent.children[index]
        if node is not None:
            return node.Q + Node.C * node.p * math.sqrt(parent.N) / (node.N+1)
        else:
            return Node.C * parent.children_p[index] * math.sqrt(parent.N)

    def rank_moves(self, node):
        """Turn the aggregate statistics from the tree into scores at the root of the mcts tree
        This uses the 'robust' way of doing this, using the visit count for each child"""
        ranks = np.zeros(node.state.action_space_size())
        for i, child in enumerate(node.children):
            if child is not None:
                rank_index = node.state.move_to_move_index(node.moves_to_children[i])
                ranks[rank_index] = child.N
        return ranks


    def get_playing_move(self, explore_temp=2):
        """
        :param parent_node: where moves are calculated from
        :param explore_temp: Controls the willingness to explore similar to softmax scaling
        :return: node after the calculated move, the used probabilities - based on the mcts search, different from nns output
        """
        ranks = self.rank_moves(self.root_node).astype(float)
        if explore_temp < MINIMUM_TEMPERATURE_ACCEPTED:
            explore_temp = 0
        else:
            ranks = ranks ** explore_temp
            ranks /= ranks.sum()
        if self.learning and explore_temp > 0:
            move_index = int(np.random.choice(np.arange(ranks.size), 1, p=ranks))
            scaled_ranks = ranks
        else:
            move_index = ranks.argmax()
            scaled_ranks = np.zeros(ranks.size)
            scaled_ranks[move_index] = 1.
        return move_index, scaled_ranks

    def backpropagation(self, node):
        v = node.V
        while not node is None:
            parent_turn = node.state.turn() if node.parent is None else node.parent.state.turn()
            partition = (int(parent_turn==node.state.turn())-0.5)*2
            v *= partition
            node.update_values(v)
            node = node.parent


    def stats(self):
        if self.search_time == 0:
            self.search_time = 0.001
        inf = {}
        inf['max_depth'] = self.root_node.max_depth()
        inf['nn_value'] = self.root_node.V
        inf['win_chance'] = self.root_node.Q/2.+0.5
        inf['n'] = self.root_node.N
        inf['time (s)'] = self.search_time
        inf['node/s'] = self.root_node.N / self.search_time
        inf['children_p'] = self.root_node.children_p
        inf['ranks'] = self.rank_moves(self.root_node).astype(int).tolist()
        return inf






