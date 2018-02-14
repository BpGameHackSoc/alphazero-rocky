import datetime
import numpy as np
import math
from src.general_player import Player

from importlib import reload
import src.tree_node
reload(src.tree_node)

from src.tree_node import Node

class MCTS():
    def __init__(self, model, **kwargs):
        self.seconds = kwargs.get('time', 3)
        self.learning = kwargs.get('learning', False)
        self.model = model
        self.time_limit = datetime.timedelta(seconds=self.seconds)

    def search(self, **kwargs):
        limit = kwargs.get('limit', None)
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
        if simulation_limit is None:
            begin = datetime.datetime.utcnow()
            while datetime.datetime.utcnow() - begin < self.time_limit: #TODO tread this
                self.run_one_simulation()
        else:
            for i in range(simulation_limit):
                self.run_one_simulation()
        return self.root_node

    def run_one_simulation(self):
        last_node = self.simulate_to_leaf()
        self.backpropagation(last_node)

    def simulate_to_leaf(self):
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
        val_moves = node.state.valid_moves()
        scores = np.array([self.UCT_score(node, move) for move in val_moves])
        best_index = np.argmax(scores)
        return val_moves[best_index]

    def UCT_score(self, parent, index):
        node = parent.children[index]
        if node is not None:
            q = node.Q if node.state.turn() == Player.A else -node.Q
            return q + Node.C * node.p * math.sqrt(parent.N) / (node.N+1)
        else:
            return Node.C * parent.children_p[index] * math.sqrt(parent.N)

    def rank_moves(self, node):
        ranks = np.zeros(len(node.children))
        for i, child in enumerate(node.children):
            if not child is None:
                ranks[i] = child.N
        return ranks

    def get_playing_move(self, parent_node, explore_temp=2):
        """

        :param parent_node: where moves are calculated from
        :param explore_temp: Controls the willingness to explore similar to softmax scaling
        :return: node after the calculated move
        """
        ranks = self.rank_moves(self.root_node).astype(float)
        ranks = ranks ** explore_temp
        ranks /= ranks.sum()
        if self.learning:
            move_index = int(np.random.choice(range(len(ranks),0), 1, p=ranks))
        else:
            move_index = ranks.argmax()
        return parent_node.get_child(move_index)

    def backpropagation(self, node):
        v = node.V
        while not node is None:
            node.update_values(v)
            node = node.parent


    def stats(self):
        inf = {}
        inf['max_depth'] = self.root_node.max_depth()
        inf['nn_value'] = self.root_node.V * -1
        inf['mcts_value'] = self.root_node.Q * -1
        inf['n'] = self.root_node.N
        inf['node/s'] = self.root_node.N / self.seconds
        inf['ranks'] = self.rank_moves(self.root_node).astype(int).tolist()
        return inf






