import numpy as np
import math
import threading

class ThreadException(Exception):
    pass





def config(self):
    value, probabilities = nn.predict(inp)
    self.V = value.flatten()[0]
    self.N = 1
    self.children_p = probabilities.flatten()
    self.children = [None] * self.children_p.size

use_terminal_score = False
class Node(object):
    """
    Responsible for keeping the data of tree node for Monte-Carlo Tree Search
    """

    C = math.sqrt(2)

    def __init__(self, state, p=0, parent=None):
        self.state = state
        self.parent = parent
        self.W = 0                  # Accumulated value of all nodes below this one
        self.V = None               # Value of this state
        self.N = 0                  # The number of visits
        self.p = p                  # The probability of choosing this node from parent
        self.Q = 0
        self.children = [None] * state.action_space_size() # The child-states  TODO replace with config value
        self.children_p = []        # The probability distribution on child states
        self.is_terminal = False

    def max_depth(self):
        if self.state.is_over() or self.is_leaf():
            return 0
        else:
            m = 0
            for child in self.children:
                if child is None:
                    continue
                current = child.max_depth()
                if current > m:
                    m = current
            return m + 1

    def define_child(self, move):
        new_state = self.state.move(move)
        self.children[move] = Node(new_state, p=self.children_p[move], parent=self)
        return self.children[move]

    def expand_and_evaluate(self,move,model):
        new_node = self.define_child(move)
        new_node.evaluate(model)
        return new_node

    def evaluate(self, model):
        if self.V is None:
            parent_turn = self.parent.state.turn() if self.parent else 0
            v, p = self.__get_prediction(model)
            self.V = v.flatten()[0] 
            self.children_p = p
            if self.state.is_over():
                self.is_terminal = True
                if use_terminal_score:
                    self.V = self.get_terminal_score()

    def __get_prediction(self, model):
        s = self.state.to_input()
        v, p = model.predict(np.expand_dims(s, axis=0))
        return v.flatten()[0], p.flatten()

    def is_leaf(self):
        return not any(self.children)

    def update_values(self, z_estimate):
        self.N += 1
        self.W += z_estimate
        self.Q = self.W / self.N

    def is_visited(self):
        return self.N!=0

    def get_aggragate_values(self):
        return (self.W, self.Q, self.N)

    def get_children_data(self):
        results = []
        for child in self.children:
            if child is None:
                results.append(None)
                continue
            results.append(child.get_aggragate_values())
        return results

    def get_terminal_score(self):
        if self.state.winner() == 0: # TODO use an abstract class for comparison
            return 0
        return -1 # assuming that current player always loses in terminal state


class Node_threaded(Node):
    """
    constant members: state, parent
    """
    def __init__(self, state, p=0, parent=None):
        super().__init__(state, p, parent)
        self.update_lock = threading.RLock()
        # self.creation_lock = threading.Lock()

    # def update_values(self, z_estimate):
    #     if self.update_lock.acquire():
    #         super().update_values(z_estimate)
    #         self.update_lock.release()
    #     else:
    #         raise ThreadException("Could not acquire lock")











