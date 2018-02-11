import numpy as np

class Node(object):

    def __init__(self, state, nn, p, parent=None):
        self.state = state
        self.parent = parent
        self.N = 0                  # The number of visits
        self.V = 0                  # Value of this state
        self.p = p                  # The probability of choosing this node from parent
        self.children = []          # The child-states
        self.children_p = []        # The probability distribution on child states
        self.config()

    def config(self):
        value, probabilities = nn.predict(inp)
        self.V = value.flatten()[0]
        self.N = 1
        self.children_p = probabilities.flatten()
        self.children = [None] * self.children_p.size

    def define_child(self, move, nn):
        new_state = self.state.move(move)
        self.children[move] = Node(new_state, nn, self.children_p[move], self)
        