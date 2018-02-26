from src.games.connect4.config import *
from src.games.connect4.color import Color
from src.general_game import GameState
from src.general_player import Player
import numpy as np

class Connect4State(GameState):
    '''
        This object represents a state of NxN gomoku table,
        where the players needs to connect M stones in order to win.
        A state is represented by [board, turn].
    '''
    def __init__(self, state=None):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        self.turn_color = Color.X       # TODO change awkward name, but not to turn because member hiddes method when named the same
        self.__winner = None

    def __str__(self):
        board = np.flip(self.board, axis=0)
        str_board = np.asarray(['-'] * self.board.size).reshape(board.shape)
        str_board[board == 1] = 'X'
        str_board[board == -1] = 'O'
        out=''
        for row in str_board:
            out += (' '.join(row) + str('\n'))
        out = out[:-1]
        out += ' Turn: ' + str(self.turn_color.str()) + str('\n')
        return out

    def turn(self):
        return Player(self.turn_color.value)

    def move_count(self):
        return (self.board != 0).sum()

    def valid_moves(self):
        return np.array(np.where(self.board[BOARD_SIZE-1] == 0)).flatten()

    def move(self, move):
        if not move in self.valid_moves():
            raise Exception('Invalid move was made: ' + str(move))
        new_state = self.copy(swap=True)
        location = np.argmax(self.board[:,move]==0)
        new_state.board[location,move] = self.turn_color.value
        return new_state

    def is_over(self):
        return self.valid_moves().size == 0 or not self.winner() == Color.NONE

    def winner(self):
        if not self.__winner is None:
            return self.__winner
        chopped_board = self.__crop()
        padded_board = self.__pad(chopped_board)
        for row in range(padded_board.shape[0] - CONNECT_SIZE + 1):
            for column in range(padded_board.shape[1] - CONNECT_SIZE + 1):
                current_chop = padded_board[row:row+CONNECT_SIZE,column:column+CONNECT_SIZE]
                winner = self.__scan(current_chop)
                if not winner == Color.NONE:
                    self.__winner = winner
                    return winner
        return Color.NONE


    def __crop(self):
        if self.__board_is_empty():
            return self.board
        true_points = np.argwhere(self.board)
        top_left = true_points.min(axis=0)
        bottom_right = true_points.max(axis=0)
        return self.board[top_left[0]:bottom_right[0]+1,
                          top_left[1]:bottom_right[1]+1]

    def __board_is_empty(self):
        return (self.board == 0).all().all()

    def __pad(self, small_board):
        size = max(CONNECT_SIZE-small_board.shape[0],
                   CONNECT_SIZE-small_board.shape[1])
        if size < 0:
            return small_board
        return np.pad(small_board, [0, size], 'constant')

    def __scan(self, board):
        '''
            Here we only consider a CONNECT_SIZE x CONNECT_SIZE sub-board and determine
            if any colors have won.
        '''
        left_diagonal_step = CONNECT_SIZE - 1
        right_diagonal_step = CONNECT_SIZE + 1
        s1 = board.sum(axis=0)
        s2 = board.sum(axis=1)
        s3 = board.reshape(-1)[CONNECT_SIZE-1:CONNECT_SIZE**2-CONNECT_SIZE+1:left_diagonal_step].sum()
        s4 = board.reshape(-1)[0:CONNECT_SIZE**2:right_diagonal_step].sum()
        s = np.concatenate([s1,s2,[s3],[s4]])
        if (s == CONNECT_SIZE * Color.X.value).any():
            return Color.X
        elif (s == CONNECT_SIZE * Color.O.value).any():
            return Color.O
        else:
            return Color.NONE


    def action_space_size(self):
        return BOARD_SIZE

    def to_all_symmetry_input(self, p):
        normal = [
            self.to_input(self.board),
            self.turn_color.value,
            p.copy()
        ]
        flipped = [
            self.to_input(np.flip(self.board,axis=1)),
            self.turn_color.value,
            np.flip(p, axis=0).copy()
        ]
        return [normal, flipped]


    def copy(self, swap=False):
        s = Connect4State()
        s.board = self.board.copy()
        s.turn_color = Color(self.turn_color.value)
        if swap:
            s.swap_turn()
        return s

    def swap_turn(self):
        self.turn_color = Color(self.turn_color.value * -1)

    def to_input(self, board=None):
        board = self.board if board is None else board
        x = np.zeros(shape=(2, BOARD_SIZE, BOARD_SIZE))
        x[0][board == self.turn_color.value] = 1
        x[1][board == -1 * self.turn_color.value] = 1
        return x
