import abc

class GameState(abc.ABC):
    '''
        The game state object represents the model of a game, the rules,
        the board and its pieces. Note it only represents one specific state
        where the game is at, but not more.

        Some MUST rules:
            1. This is a zero-sum, 2 player game
            2. The moves are integers, starting from 0
    '''
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def __str__(self):
        '''
            A string representation of the state
        '''
        pass

    @abc.abstractmethod
    def turn(self):
        '''
            Since each game the engine solves is a two-player zero-sum game,
            each state must be someone's turn. This function should return 
            with the Player enum object.
        '''
        return Player(1)

    @abc.abstractmethod
    def move_count(self):
        '''
            Returns with how many moves were made so far during the game.
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
    def action_space_size(self):
        '''
            Returns with the number of possible moves
            (excluding the fact if they are valid or not)
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
    def copy(self):
        '''
            Returns a copy of this object
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

    @abc.abstractmethod
    def to_input(self):
        '''
            Converts the state to an input for a neural network.
            E.g. in chess, it should handle the following information:
                [board, turn, castling_rights, en_passant, half_moves]

            Note this function must return with a numpy matrix as:
                [state, (int) player_on_turn, probabilities]

            The idea behind this method is that the board in some games such as othello
            or go can be be rotated or flipped.

        '''
        pass

    @abc.abstractmethod
    def to_all_symmetry_input(self, probabilities):
        '''
            Returns with the same as to_input, but all possible rotations & flips,
            along with flipping the move probabilities. Output should look like:
               [[state, (int) player_on_turn, probabilities]
                [state, (int) player_on_turn, probabilities]
                [state, (int) player_on_turn, probabilities]
                [state, (int) player_on_turn, probabilities]]
        '''
        pass