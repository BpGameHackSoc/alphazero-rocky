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

use_terminal_score = True
class Node(object):
    """
    Responsible for keeping the data of tree node for Monte-Carlo Tree Search
    """

    def define_child(self, move):
        new_state = self.state.move(move)
        self.children[move] = type(self)(new_state, p=self.children_p[move], parent=self)
        return self.children[move]

    def expand_and_evaluate(self,move,model):
        new_node = self.define_child(move)
        new_node.evaluate(model)
        return new_node

    C = math.sqrt(2)

    def __init__(self, state, p=0, parent=None):
        self.state = state
        self.parent = parent
        self.W = 0                  # Accumulated value of all nodes below this one
        self.V = 0                  # Value of this state
        self.N = 0                  # The number of visits
        self.p = p                  # The probability of choosing this node from parent
        self.Q = 0
        self.children = [None] * state.action_space_size() # The child-states  TODO replace with config value
        self.children_p = []        # The probability distribution on child states

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

    def evaluate(self, model, back_prop_real=False):
        is_game_over = self.state.is_over()
        if is_game_over and use_terminal_score:
            z = self.get_terminal_score()
            self.update_values(z)
            self.is_terminal = True
        else:
            s = self.state.to_input()
            v, p = model.predict(np.expand_dims(s, axis=0))
            v = v.flatten()
            self.V = v
            self.update_values(v)
            if not is_game_over:
                self.children_p = p.flatten()


    def is_leaf(self):
        return not any(self.children)

    def get_child(self,move):
        return self.children[move]

    def get_Q(self):
            return self.Q

    def get_p(self):
        return self.p

    # def set_V(self,v):
    #     if self.parent is None:
    #         pass
    #     self.V = v

    def get_N(self):
        return self.N

    def update_values(self, z_estimate):
        self.__increment_N()
        self.__add_to_W(z_estimate)
        self.calc_Q()

    def get_V(self):
        return self.V


    def calc_Q(self):
        self.Q = self.W / self.N

    def __increment_N(self):
        self.N += 1

    def is_terminal(self):
        return self.state.is_over()

    def __add_to_W(self, x):
        self.W += x

    def is_visited(self):
        return self.get_N()!=0

    def get_aggragate_values(self):
        return (self.W, self.Q, self.N)

    def get_children_data(self):
        result = [child.get_aggragate_values for child in self.children]
        return result

    def get_terminal_score(self):
        if self.state.winner == 0: # TODO use an abstract class for comparison
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











