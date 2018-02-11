from src.general_agent import Agent
import numpy as np

class RandomAgent(Agent):
    '''
        This agent plays random moves by choosing from the valid ones.
    '''

    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'random')

    def move(self, state):
        valid = state.valid_moves()
        return np.random.choice(valid)

    def evaluate(self, state):
        if not state.is_over():
            print('Random from ' + str(state.valid_moves()))

