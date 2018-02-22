from src.games.gomoku3d.config import *
from src.games.gomoku3d.color import Color
from src.general_game import GameState
from src.general_player import Player
import numpy as np

class Gomoku3dState(GameState):
    '''
        This object represents a state of NxN gomoku table,
        where the players needs to connect M stones in order to win.
        A state is represented by [board, turn].
    '''
    def __init__(self, state=None):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE, BOARD_SIZE))
        self.turn_color = Color.BLUE       # TODO change awkward name, but not to turn because member hiddes method when named the same
        self.__winner = None

    def __str__(self):
        str_board = np.asarray(['-'] * self.board.size).reshape(self.board.shape)
        str_board[self.board == 1] = 'X'
        str_board[self.board == -1] = 'O'
        str_board = np.rot90(str_board)
        out=''
        for dimension in reversed(str_board):
            for row in dimension:
                out += (' '.join(row) + str('   '))
            out += str('\n')
        out = out[:-1]
        out += ' Turn: ' + str(self.turn_color.str()) + str('\n')
        return out

    def turn(self):
        return Player(self.turn_color.value)

    def move_count(self):
        return (self.board != 0).sum()

    def valid_moves(self):
        return np.array(np.where(self.board[BOARD_SIZE-1].reshape(-1) == 0)).flatten()

    def move(self, move):
        if not move in self.valid_moves():
            raise Exception('Invalid move was made: ' + str(move))
        new_state = self.copy(swap=True)
        row = move // 4
        column = move % 4
        location = np.argmax(self.board[:,row,column]==0)
        new_state.board[location,row,column] = self.turn_color.value
        return new_state

    def is_over(self):
        return self.valid_moves().size == 0 or not self.winner() == Color.NONE

    def winner(self):
        if not self.__winner is None:
            return self.__winner
        n = BOARD_SIZE * BOARD_SIZE
        layers = self.get_layers()
        sum1 = layers[:].sum(axis=1).flatten()
        sum2 = layers[:].sum(axis=2).flatten()
        sum3 = layers.reshape(-1,n)[:,BOARD_SIZE-1:n-BOARD_SIZE+1:BOARD_SIZE-1].sum(axis=1)
        sum4 = layers.reshape(-1,n)[:,0:n:BOARD_SIZE+1].sum(axis=1)
        s = np.concatenate((sum1, sum2, sum3, sum4))
        if (s == CONNECT_SIZE).any():
            self.__winner = Color.BLUE
            return self.__winner
        elif (s == -CONNECT_SIZE).any():
            self.__winner = Color.RED
            return self.__winner
        return Color.NONE

    def get_layers(self):
        n = BOARD_SIZE*BOARD_SIZE
        x = np.zeros((3*BOARD_SIZE+2,BOARD_SIZE,BOARD_SIZE), dtype=int)
        for dim in range(3):
            for j in range(BOARD_SIZE):
                i = dim * BOARD_SIZE + j
                if dim == 0:
                    x[i] = self.board[j,:,:]
                elif dim == 1:
                    x[i] = self.board[:,j,:]
                else:
                    x[i] = self.board[:,:,j]
        x[-2] = self.board.reshape(-1,n)[:,0:n:BOARD_SIZE+1].reshape(4,4)
        x[-1] = self.board.reshape(-1,n)[:,BOARD_SIZE-1:n-BOARD_SIZE+1:BOARD_SIZE-1].reshape(4,4)
        return x

    def action_space_size(self):
        return BOARD_SIZE * BOARD_SIZE

    def to_all_symmetry_input(self, p):
        transformations = []
        p = p.reshape(BOARD_SIZE, BOARD_SIZE)
        for i in range(4):
            rotated_board = self.board_rotation(i).copy()
            rotated_p = np.rot90(p, i).copy()
            one_input = [self.to_input(rotated_board),
                         self.turn_color.value, rotated_p.reshape(-1)]
            transformations.append(one_input)
            flipped_board = np.flip(rotated_board, axis=1).copy()
            flipped_p = np.flip(rotated_p, axis=0).copy()
            one_input = [self.to_input(flipped_board),
                         self.turn_color.value, flipped_p.reshape(-1)]
            transformations.append(one_input)
        return transformations

    def board_rotation(self, half_radians):
        return np.rot90(self.board, half_radians, axes=(-2,-1))

    def copy(self, swap=False):
        s = Gomoku3dState()
        s.board = self.board.copy()
        s.turn_color = Color(self.turn_color.value)
        if swap:
            s.swap_turn()
        return s

    def swap_turn(self):
        self.turn_color = Color(self.turn_color.value * -1)

    def to_input(self, board=None):
        board = self.board if board is None else board
        x = np.zeros(shape=(BOARD_SIZE*2, BOARD_SIZE, BOARD_SIZE))
        x[:BOARD_SIZE][board == self.turn_color.value] = 1
        x[BOARD_SIZE:][board == -1 * self.turn_color.value] = 1
        return x
