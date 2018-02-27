from src.general_agent import Agent
import numpy as np
from src.mcts import MCTS
from src.config import MINIMUM_TEMPERATURE_ACCEPTED, DEFAULT_TRAIN_THINK_TIME


class StudentAgent(Agent):
    '''
        This agent plays moves driven by a probability distribution
    '''

    def __init__(self, neural_net, **kwargs):
        self.name = kwargs.get('name', 'student')
        self.learning = kwargs.get('learning', True)
        self.think_time = kwargs.get('think_time', DEFAULT_TRAIN_THINK_TIME)
        self.nn = neural_net
        self.mcts = MCTS(neural_net, learning=self.learning, think_time=self.think_time)
        self.last_run = {}

    def move(self, state, **kwargs):
        temp = kwargs.get('temp', 0)
        pre_known_node = kwargs.get('root', None)
        root = self.mcts.search(root=pre_known_node, state=state)
        visit_counts = self.mcts.rank_moves(root)
        probabilities = visit_counts / float(visit_counts.sum())
        move = self.mcts.get_playing_move(temp)
        self.last_run['stats'] = self.mcts.stats()
        self.last_run['probabilities'] = probabilities
        self.last_run['chosen_child'] = root.children[move]
        return move

    def evaluate(self, state, **kwargs):
        if not state.is_over():
            print('Valid moves:' + str(state.valid_moves()))
            print(self.last_run['stats'])

    def calculate_real_distribution(self, visit_count_distribution, temp):
        distribution = visit_count_distribution ** temp
        distribution = distribution / distribution.sum()
        return distribution