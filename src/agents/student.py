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
        move_index, move_index_in_valid,  probabilities = self.mcts.get_playing_move(temp)
        move = state.move_index_to_move(move_index)
        self.last_run['stats'] = self.mcts.stats()
        self.last_run['probabilities'] = probabilities
        self.last_run['chosen_child'] = root.children[move_index_in_valid]
        self.last_run['confidence'] = root.children[move_index_in_valid].N / root.N
        self.last_run['predicted_outcome'] = root.Q
        self.last_run['last_move'] = move
        self.last_run['move_index_in_valid'] = move_index_in_valid
        return move

    def evaluate(self, state, **kwargs):
        if not state.is_over():
            print('Valid moves:' + str(state.valid_moves()))
            print(self.str_stats())

    def calculate_real_distribution(self, visit_count_distribution, temp):
        distribution = visit_count_distribution ** temp
        distribution = distribution / distribution.sum()
        return distribution

    def str_stats(self):
        s = self.last_run['stats']
        move_index_in_valid = self.last_run['move_index_in_valid']

        out = '-' * 80 + '\n'
        out += '| Simulations: %13d | Time (s): %13.2f | Node/s: %13.2f |\n' % (s['n'], s['time (s)'], s['node/s'])
        out += '-' * 80 + '\n'
        out += '| children_p: %-65s|\n' % s['children_p'].round(2).tolist()
        out += '-' * 80 + '\n'
        out += '| Visits: %-69s|\n' % s['ranks']
        out += '-' * 80 + '\n'
        out += '| NN value: %16.2f | Win chance: %10.2f%% | Max depth: %10d |\n' % (s['nn_value'],
                                                                                 s['win_chance'] * 100,
                                                                                 s['max_depth'])
        out += '=' * 80 + '\n'
        out += '| Preferred move: %-20d | Final move: %-26d|\n' % (s['children_p'].argmax(), move_index_in_valid)
        out += '-' * 80 + '\n'
        return out
