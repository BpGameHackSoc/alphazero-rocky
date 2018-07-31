from src.general_game import GameState
from src.general_player import Player
from .errors import MoveError,BoardNotFinishedError,MoveInFinishedBoardError
import numpy as np
import copy

Player
class UTTTState(GameState):

    BOARD_SIZE = 3
    MAIN_DIAG_INDEXES = np.array([[0,0],[1,1],[2,2]])
    MINOR_DIAG_INDEXES = np.array([[2,0],[1,1],[0,2]])
    PLAYER_ARRAY = [0, 0, 0]
    PLAYER_ARRAY[Player.FIRST] = np.full((BOARD_SIZE,),Player.FIRST)
    PLAYER_ARRAY[Player.SECOND] = np.full((BOARD_SIZE,), Player.SECOND)

    def __init__(self):
        self.board = np.zeros((self.BOARD_SIZE,)*4)
        self.global_wins = np.zeros((self.BOARD_SIZE,) * 2)
        self.curr_player = Player.FIRST
        self.moves_played = 0
        self.active_subgame = None
        self.won_by = Player.NONE

    def __str__(self):
        """Returns a pretty printed representation of the main board"""
        pretty_printed = ''
        # TODO: Shouldn't access sub-board private var
        board_size = len(self.board)
        board_size_range = range(board_size)
        for (mb_idx, mb_row) in enumerate(self.board):
            for sub_board_row_num in board_size_range:
                for (sb_idx, sub_board) in enumerate(mb_row):
                    for cell in sub_board[sub_board_row_num]:
                        pretty_printed += str(int(cell)) + ' '
                    # Print vertical separator - if not last sub_board
                    if sb_idx < board_size - 1:
                        pretty_printed += '| '
                pretty_printed += '\n'
            # Print horizontal separators
            # Only if this is not the last row
            if mb_idx < board_size - 1:
                for (bm_idx, board_marker) in enumerate(board_size_range):
                    for cell_marker in board_size_range:
                        pretty_printed += '- '
                    if bm_idx < board_size - 1:
                        pretty_printed += '| '
                pretty_printed += '\n'

        return pretty_printed

    def turn(self) -> Player:
        return self.curr_player

    def move_count(self) -> int:
        return self.moves_played

    def valid_moves(self):
        if self.active_subgame is None:
            good_pos = self.board==Player.NONE
            good_pos[self.global_wins != 0] = False
            return np.argwhere(good_pos)
        else:
            active_board = self.board[self.active_subgame]
            return np.argwhere(active_board==Player.NONE)

    def apply_move_to_board(self,move,player):
        self.board[move] = player

    def apply_submove_to_subboard(self,submove, subboard,player):
        subboard[submove] = player

    def is_forbidden_submove(self, submove, subgame) -> bool:
        return subgame[submove]

    def move(self, move):
        other = copy.deepcopy(self)
        other.move_in_place(move)
        return other

    def move_in_place(self, move):
        """Apply move to the board. Move is a 4 dimensional tuple index if self.active_subgame if None,
        2 dimensional otherwise."""
        move = tuple(move)
        if self.is_over():
            raise MoveInFinishedBoardError
        player = self.curr_player
        if self.active_subgame is None:
            submove = move[2:]
            main_index = move[:2]
            subgame = self.board[move[0], move[1]]
        else:
            submove = move
            main_index = self.active_subgame
            subgame = self.board[main_index]
        if self.is_forbidden_submove(submove, subgame):
            raise MoveError
        self.apply_submove_to_subboard(submove,subgame,player)
        sub_won = self.game_3x3_won_by(subgame, submove , player)
        if sub_won:
            self.record_sub_finishes(main_index, sub_won)
            main_won = self.game_3x3_won_by(self.global_wins, main_index, player)
            if main_won:
                self.record_game_finishes(main_won)
        self.manage_active_subgame(submove)
        self.swtich_player()
        self.moves_played +=1

    def get_sub_res_by_index(self,main_index):
        return self.global_wins[main_index]

    def sub_finished(self,main_index):
        return self.get_sub_res_by_index(main_index)

    def manage_active_subgame(self, main_index):
        if self.sub_finished(main_index):
            self.active_subgame = None
        else:
            self.active_subgame = main_index

    def game_3x3_won_by(self, board_3x3, move_3x3, player):
        if self.did_submove_win_subgame(move_3x3,board_3x3,player):
            return player
        if self.subgame_full(board_3x3):
            return Player.DRAW
        return  Player.NONE

    def subgame_full(self,subgame):
        return Player.NONE not in subgame

    def record_game_finishes(self, result):
        self.won_by = result

    def record_sub_finishes(self,main_index, player):
        self.global_wins[main_index] = player

    def did_move_end_subgame(self,move,player):
        subgame = self.board[move[0], move[1]]
        return self.did_submove_win_subgame(move[2:], subgame, player)


    def did_submove_win_subgame(self, submove, subgame, player):
        winning_array = UTTTState.PLAYER_ARRAY[player]
        return np.array_equal(self.get_main_diag(subgame),winning_array) or \
                   np.array_equal(self.get_minor_diag(subgame),winning_array) or \
                   np.array_equal(self.get_col_from_submove(subgame,submove), winning_array) or \
                   np.array_equal(self.get_row_from_submove(subgame,submove), winning_array)

    def get_row_from_submove(self, subgame, submove):
        return subgame[:, submove[1]]

    def get_col_from_submove(self, subgame, submove):
        return subgame[submove[0], :]

    def get_main_diag(self,subboard):
        return subboard[self.MAIN_DIAG_INDEXES]

    def get_minor_diag(self,subboard):
        return subboard[self.MINOR_DIAG_INDEXES]

    def all_elements_equal(self, arr, axis=0):
        return np.all(arr == arr[axis])

    def is_over(self):
        return self.won_by

    def winner(self):
        if self.won_by:
            return Player(self.won_by)
        else:
            raise BoardNotFinishedError

    def get_layers(self):
        pass
        # n = BOARD_SIZE*BOARD_SIZE
        # x = np.zeros((3*BOARD_SIZE+2,BOARD_SIZE,BOARD_SIZE), dtype=int)
        # for dim in range(3):
        #     for j in range(BOARD_SIZE):
        #         i = dim * BOARD_SIZE + j
        #         if dim == 0:
        #             x[i] = self.board[j,:,:]
        #         elif dim == 1:
        #             x[i] = self.board[:,j,:]
        #         else:
        #             x[i] = self.board[:,:,j]
        # x[-2] = self.board.reshape(-1,n)[:,0:n:BOARD_SIZE+1].reshape(4,4)
        # x[-1] = self.board.reshape(-1,n)[:,BOARD_SIZE-1:n-BOARD_SIZE+1:BOARD_SIZE-1].reshape(4,4)
        # return x

    def action_space_size(self):
        return self.board.size

    def to_all_symmetry_input(self, p):
        # transformations = []
        # p = p.reshape(BOARD_SIZE, BOARD_SIZE)
        # for i in range(4):
        #     rotated_board = self.board_rotation(i).copy()
        #     rotated_p = np.rot90(p, i).copy()
        #     one_input = [self.to_input(rotated_board),
        #                  self.turn_color.value, rotated_p.reshape(-1)]
        #     transformations.append(one_input)
        #     flipped_board = np.flip(rotated_board, axis=1).copy()
        #     flipped_p = np.flip(rotated_p, axis=0).copy()
        #     one_input = [self.to_input(flipped_board),
        #                  self.turn_color.value, flipped_p.reshape(-1)]
        #     transformations.append(one_input)
        # return transformations
        pass

    def board_rotation(self, half_radians):
        return np.rot90(self.board, half_radians, axes=(-2,-1))

    def copy(self, swap=False):
        pass

    def swtich_player(self):
        self.curr_player = self.curr_player * -1

    def to_input(self, board=None):
        pass
