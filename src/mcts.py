import datetime
import numpy as np
import math
from src.general_player import Player
from src.tree_node import Node,Node_threaded




class MCTS():
    def __init__(self, model, **kwargs):
        self.seconds = kwargs.get('time', 3)
        self.learning = kwargs.get('learning', False)
        self.model = model
        self.time_limit = datetime.timedelta(seconds=self.seconds)

    def search(self, root_node, simulation_limit=None):
        self.root_node = root_node
        self.root_node.parent = None
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
        last_node, nodes_visited = self.simulate_to_leaf()
        self.backpropagation(last_node,nodes_visited)

    def simulate_to_leaf(self):
        nodes_visited = []
        node = self.root_node
        while not node.is_terminal():
            nodes_visited.append(node)
            move_index = self.select(node)
            new_node = node.children[move_index]
            if new_node is None:
                node = node.expand_and_evaluate(move_index,self.model)
                break
            node = new_node
        return (node,nodes_visited)


    def select(self, node):
        side = node.state.turn()
        scores = np.array([self.UCT_score(child,i,node) for i,child in enumerate(node.children)])
        if side == Player.A:
            best_index = np.argmax(scores)
        else:
            best_index = np.argmin(scores)
        return best_index

    def UCT_score(self, node, index, parent):
        if node is not None:
            return node.get_Q() + Node.C * node.get_p() * math.sqrt(parent.get_N()) / (node.get_N()+1)
        else:
            return Node.C * parent.children_p[index] * math.sqrt(parent.get_N())

    def rank_moves(self, node):
        ranks = np.zeros(len(node.children))
        for i, child in enumerate(node.children):
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

    def backpropagation(self, node,nodes_visited):
        v = node.get_V()
        for node_inner in nodes_visited:
            node_inner.update_values(v)


    # def stats(self):
    #     inf = {}
    #     inf['max_depth'] = self.root_node.max_depth()
    #     inf['prediction'] = self.root_node.UCT_score(self.root_node.gomoku.turn)
    #     inf['blue_q'] = self.root_node.UCT_score(Color.BLUE)
    #     inf['red_q'] = self.root_node.UCT_score(Color.RED)
    #     inf['n'] = self.root_node.N
    #     inf['node/s'] = self.root_node.N / self.seconds
    #     ranks = self.rank_moves(self.root_node)
    #     ranks = ranks[ranks[:, 1].argsort(kind='mergesort')]
    #     ranks = ranks[::-1]
    #     inf['ranks'] = ranks
    #     return inf


class MCTS_threaded(MCTS):
    def __init__(self, model, **kwargs):
        super().__init(model, **kwargs)
