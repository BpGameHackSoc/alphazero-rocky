import datetime
import numpy as np
import math
from src.tree_node import Node,Node_threaded
class MCTS_threaded(MCTS):
    def __init__(self, model, **kwargs):
        super().__init(model, **kwargs)


class MCTS():
    def __init__(self, model, **kwargs):
        self.seconds = kwargs.get('time', 3)
        self.learning = kwargs.get('learning', False)
        self.model = model
        self.time_limit = datetime.timedelta(seconds=self.seconds)

    def search(self, root_node):
        self.root_node = root_node
        self.root_node.parent = None
        begin = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - begin < self.time_limit: #TODO tread this
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
                return node.expand_and_evaluate(move_index,self.model)
            node = new_node
        return node


    def select(self, node):
        side = node.state.player
        scores = np.array([self.UCT_score(child,node) for child in node.children])
        best_index = scores.argmax(scores)
        return best_index

    def UCT_score(self, node,parent):
        return node.get_Q() + Node.C * node.get_p() * math.sqrt(parent.get_N) / node.get_N()

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

    def backpropagation(self, node):
        current_node = node.parent
        while not current_node is None:
            current_node.Q_blue += node.get_Q(Color.BLUE)
            current_node.Q_red += node.get_Q(Color.RED)
            current_node.N += 1
            current_node = current_node.parent

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


