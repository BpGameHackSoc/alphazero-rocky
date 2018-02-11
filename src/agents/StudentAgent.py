from src.general_agent import Agent
import numpy as np
from src.config import MINIMUM_TEMPERATURE_ACCEPTED, DEFAULT_TRAIN_THINK_TIME

class StudentAgent(Agent):
    '''
        This agent plays moves driven by a probability distribution
    '''

    def __init__(self, neural_net, **kwargs):
        self.name = kwargs.get('name', 'student')
        self.nn = neural_net
        self.think_time = kwargs.get('think_time', DEFAULT_TRAIN_THINK_TIME)
        self.mcts = MCTS(neural_net, think_time=think_time)
        self.last_run = {}

    def move(self, state, **kwargs):
        temp = kwargs.get('temp', 0)
        pre_known_node = kwargs.get('root', None)
        root = self.mcts.search(pre_known_node)
        visit_counts = self.mcts.visit_counts(root)
        probabilities = visit_counts / float(visit_counts.sum())
        valid = state.valid_moves()
        distribution = probabilities[valid_moves]
        if temp < MINIMUM_TEMPERATURE_ACCEPTED:
            return valid[distribution.argmax()]
        real_distribution = calculate_real_distribution(distribution, temp)
        move = np.random.choice(valid, p=real_distribution)

        self.last_run['probabilities'] probabilities
        self.last_run['chosen_child'] = root[move]

        return move

    def evaluate(self, state, visit_count_distribution, temp):
        if not state.is_over():
            print('Valid moves:' + str(state.valid_moves()))
            print('Visits:' + str(state.valid_moves()))
            if temp < MINIMUM_TEMPERATURE_ACCEPTED:
                print('Temp was lower than ' + str(MINIMUM_TEMPERATURE_ACCEPTED))
            else:
                valid = state.valid_moves()
                distribution = distribution[valid_moves]
                real_distribution = calculate_real_distribution(distribution, temp)
                print('Distribution: ' + real_distribution)


    def calculate_real_distribution(self, distribution, temp):
        distribution = visit_count_distribution[valid] ** temp
        distribution = distribution / distribution.sum()
        return distribution