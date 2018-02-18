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
        self.think_time = kwargs.get('think_time', DEFAULT_TRAIN_THINK_TIME)
        self.nn = neural_net
        self.mcts = MCTS(neural_net, think_time=self.think_time)
        self.last_run = {}

    def move(self, state, **kwargs):
        temp = kwargs.get('temp', 0)
        pre_known_node = kwargs.get('root', None)
        root = self.mcts.search(root=pre_known_node, state=state)
        visit_counts = self.mcts.rank_moves(root)
        probabilities = visit_counts / float(visit_counts.sum())
        valid = state.valid_moves()
        distribution = probabilities[valid]
        if temp < MINIMUM_TEMPERATURE_ACCEPTED:
            move = valid[distribution.argmax()]
            self.last_run['probabilities'] = probabilities
            self.last_run['chosen_child'] = root.children[move]
            return move
        else:
            real_distribution = self.calculate_real_distribution(distribution, temp)
            move = np.random.choice(valid, p=real_distribution)
            self.last_run['probabilities'] = probabilities
            self.last_run['chosen_child'] = root.children[move]

        return move

    def evaluate(self, state, visit_count_distribution, temp):
        if not state.is_over():
            print('Valid moves:' + str(state.valid_moves()))
            print('Visits:' + str(state.valid_moves()))
            if temp < MINIMUM_TEMPERATURE_ACCEPTED:
                print('Temp was lower than ' + str(MINIMUM_TEMPERATURE_ACCEPTED))
            else:
                valid = state.valid_moves()
                distribution = visit_count_distribution[valid]
                real_distribution = self.calculate_real_distribution(distribution, temp)
                print('Distribution: ' + real_distribution)


    def calculate_real_distribution(self, visit_count_distribution, temp):
        distribution = visit_count_distribution ** temp
        distribution = distribution / distribution.sum()
        return distribution

    def clone(self,nn_path=None):
        if nn_path is not None:
            copy_nn = type(self.nn)(nn_path)
        else:
            copy_nn = self.nn.clone()
        return StudentAgent(copy_nn,think_time=self.think_time,name=self.name+'_cl')
