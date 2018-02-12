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

class Node(object):
    """
    Responsible for keeping the data of tree node for Monte-Carlo Tree Search
    """

    def define_child(self, move):
        new_state = self.state.move(move)
        self.children[move] = Node(new_state, self.children_p[move], self)
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
        self.children = [None] * 16 # The child-states  TODO replace with config value
        self.children_p = []        # The probability distribution on child states

    # def max_depth(self):
    #     if self.state.is_over() or not self.visited():
    #         return 0
    #     else:
    #         m = 0
    #         for child in self.children:
    #             current = child.max_depth()
    #             if current > m:
    #                 m = current
    #         return m + 1

    def evaluate(self, model, back_prop_real=False):
        is_game_over = self.state.is_over()
        if is_game_over and back_prop_real:
            z = self.get_terminal_score()
            self.update_values(z)
            self.is_terminal = True
        else:
            s = self.state.to_input()
            v, p = model.predict(s)
            v = v.flatten()
            self.update_values(v)
            if not is_game_over:
                self.children_p = p.flatten()


    def is_leaf(self):
        return len(self.children) == 0

    def get_child(self,index):
        return self.children[index]

    def get_Q(self):
            return self.Q

    def get_p(self):
        return self.p

    def get_N(self):
        return self.N

    def update_values(self, z_estimate):
        self.__increment_N()
        self.__add_to_W(z_estimate)
        self.__calc_Q()
        self.V = z_estimate

    def get_V(self):
        return self.V


    def __calc_Q(self):
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
    def __init__(self, gomoku, move, parent, p):
        super().__init__(gomoku, move, parent, p)
        self.update_lock = threading.Lock()
        self.creation_lock = threading.Lock()

    def update_values(self, z_estimate):
        if self.update_lock.acquire():
            super().update_values(z_estimate)
            self.update_lock.release()
        else:
            raise ThreadException("Could not acquire lock")











