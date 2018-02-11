from src.general_agent import Agent
import numpy as np

class HumanAgent(Agent):
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'human')
    def move(self, state, **kwargs):
        print ('Please choose from: ' + str(state.valid_moves()))
        move = input()
        return int(move)
    def evaluate(self, state):
        pass