import abc

class Agent(abc.ABC):
    '''
        An Agent is a player of the game who given a state
        chooses a move to play.
    '''

    @abc.abstractmethod
    def __init__(self, name, thinking_time):
        self.name = name
        self.thinking_time = thinking_time
        
    @abc.abstractmethod
    def move(self, state):
        '''
            The agent takes a state as an input and outputs
            the best possible move can be made.
        '''
        pass
    @abc.abstractmethod
    def evaluate(self, state):
        '''
            The agent prints its thoughts about the current state
            to the standard output
        '''
        pass


