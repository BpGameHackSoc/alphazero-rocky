import abc

class GameState(abc.ABC):
    '''
        The game state object represents the model of a game, the rules,
        the board and its pieces. Note it only represents one specific state
        where the game is at, but not more.
    '''

    @abc.abstractmethod
    def __str__(self):
        '''
            A string representation of the state
        '''
        pass

    @abc.abstractmethod
    def move(self, move_index):
        '''
            This method checks if the desired move is valid.
            If so, creates a new state instance, applies the move
            and returns with the next state.
        '''
        return GameState()

    @abc.abstractmethod
    def valid_moves(self):
        '''
            Returns with all the valid moves that is possible to be made
            from this particular state.
        '''
        pass

    @abc.abstractmethod
    def is_over(self):
        '''
            Checks if the game is over.
                return: [True/False]
        '''
        pass

    @abc.abstractmethod
    def winner(self):
        '''
            Returns with the winner of the game:
                return: [IntEnum]   1 --> Starting player won
                                   -1 --> Second player won
                                    0 --> Draw
        '''
        pass