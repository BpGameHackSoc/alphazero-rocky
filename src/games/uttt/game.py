from src.general_game import GameState
from src.general_player import Player
from .errors import MoveError,BoardNotFinishedError,MoveInFinishedBoardError
import numpy as np
import copy
import tensorflow as tf


class UTTTState(GameState):

    BOARD_SIZE = 3
    MAIN_DIAG_INDEXES = ([0,1,2],[0,1,2])
    MINOR_DIAG_INDEXES = ([2,1,0],[0,1,2])
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

    def move_to_move_index(self,move):
        if self.active_subgame is None:
            move = np.transpose(move)
            return np.ravel_multi_index(move, self.board.shape)
        else:
            move = np.transpose(np.concatenate((self.active_subgame,move),axis=-1))
            return np.ravel_multi_index(move, self.board.shape)
    def move_index_to_move(self,move_index):
        if self.active_subgame is None:
            move = np.unravel_index(move_index,self.board.shape)
            return move
        else:
            move = np.unravel_index(move_index,self.board.shape)
            if move[:2] != self.active_subgame:
                raise ValueError("Illegal move index {} resultes in move {}".format(move_index,move))
            return move[2:]

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

    def get_layers(self, board, gloabal_wins, active_subgame_):
        vanilla_board = board
        subgame_results = tf.one_hot(gloabal_wins +1,3,name='subgame_resuts')
        if active_subgame_ is None:
            active_subgame = 9
        else:
            active_subgame = np.ravel_multi_index(active_subgame_,board.shape[:2])
        no_subgames = board[0,0,:,:].size
        act_sub_t = tf.one_hot(active_subgame, no_subgames+1,name='active_subgame')
        nearly_done = []
        for subgame in vanilla_board.reshape(-1,*vanilla_board.shape[2:]).copy():
            axes = np.concatenate((subgame, subgame.T,
                                   self.get_main_diag(subgame)[None, :],
                                   self.get_minor_diag(subgame)[None, :]), axis=0)
            axes[(np.isin(np.count_nonzero(axes, axis=1), (0, 3))) & (
                    np.count_nonzero(axes, axis=1) == np.absolute(np.sum(axes, axis=1)))] = 0
            axes[(np.count_nonzero(axes, axis=1) != np.absolute(np.sum(axes, axis=1)))] = 1
            pos_part = np.sum(axes, axis=1) >= 0
            non_pos_part = ~pos_part
            cont = np.zeros((8,),dtype=np.int64)
            # np.ravel_multi_index(axes[pos_part].T,(2,2,2))
            cont[pos_part] = np.ravel_multi_index(axes[pos_part].T.astype(np.int64), (2, 2, 2))
            cont[non_pos_part] = np.ravel_multi_index(axes[non_pos_part].T.astype(np.int64), (2, 2, 2), 'wrap') + 7
            cont = tf.one_hot(cont,14)
            nearly_done.append(cont)
        vanilla_board = vanilla_board.swapaxes(-2, -3).reshape((9, 9))
        vanilla_board = vanilla_board[np.newaxis,...,np.newaxis]
        threealogline = tf.reshape(tf.concat(nearly_done, axis=0), [-1])
        subgame_res = tf.reshape(subgame_results, [-1])
        active_sub = tf.reshape(act_sub_t, [-1])
        try:
            hand_features = tf.concat([threealogline, subgame_res, active_sub], axis=0)
        except ValueError:
            pass
            raise
        with tf.Session().as_default():
            hand_features = np.array(hand_features.eval())
        hand_features = hand_features[np.newaxis,...]
        return [vanilla_board,hand_features]

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
    def reshape_board_to_2D(self,board):
        new_board = board.swapaxes(-2, -3).reshape((9, 9))
        return new_board

    def action_space_size(self):
        return self.board.size

    def to_all_symmetry_input(self, p):
        try:
            val_moves = self.val_moves
        except AttributeError:
            self.val_moves = self.valid_moves()
            val_moves = self.val_moves
        p = p.reshape((3,3,3,3))
        transformations = []
        for i in range(4):
            rotated_board = self.rotate_4dim_shape(self.board,i).copy()
            rotated_glob_wins = np.rot90(self.global_wins, i, (0, 1))
            if self.active_subgame is None:
                active_subgame = None
            else:
                active_subgame = self.rotate_2D_index(self.active_subgame,i,(3,3))
            rotated_p = self.rotate_4dim_shape(p,i).copy()
            one_input = [self.get_layers(rotated_board,rotated_glob_wins,active_subgame),
                         self.turn(), rotated_p.reshape(-1)]
            transformations.append(one_input)
            rotated_board = self.flip_4dim_shape(rotated_board).copy()
            rotated_glob_wins = np.flip(rotated_glob_wins, axis=0)
            if active_subgame is not None:
                active_subgame = self.flip_2D_index(active_subgame, (3, 3))
            rotated_p = self.flip_4dim_shape(rotated_p).copy()
            one_input = [self.get_layers(rotated_board,rotated_glob_wins,active_subgame),
                         self.turn(), rotated_p.reshape(-1)]
            transformations.append(one_input)
        return transformations
        pass

    def filter_by_valid(self,nnp,valid_moves):
        """
        Filter the probability distribution given by the neural network to only contain entries that correspond to
        valid moves
        :param nnp: 1D numpy array, probabilty output of nn
        :param valid_moves: result of a self.valid_moves() call
        :return: 1D numpy array only containing values whose indices corresponded to valid moves in the original array
        """
        nnp = nnp.reshape(self.board.shape)
        if valid_moves.shape[1] == 4:
            indices = np.transpose(valid_moves)
            return nnp[tuple(indices)]
        else:
            subgame_filtered_nnp = nnp[self.active_subgame]
            sub_indices = np.transpose(valid_moves)
            return subgame_filtered_nnp[tuple(sub_indices)]

    def rotate_4dim_shape(self, board,half_radians):
        rotated_board = np.rot90(board, half_radians, (0, 1))
        rotated_board = np.rot90(rotated_board, half_radians, (2, 3))
        return rotated_board

    def flip_4dim_shape(self, board):
        rotated_p = np.flip(board, axis=0)
        rotated_p = np.flip(rotated_p, axis=-2)
        return rotated_p

    def flip_2D_index(self,index,shape):
        b = np.zeros(shape)
        b[index] = 1
        b = np.flip(b, axis=0)
        act = np.nonzero(b)
        return act

    def rotate_2D_index(self,index,half_radians,shape):
        b = np.zeros(shape)
        b[index] = 1
        b = np.rot90(b, half_radians, (0, 1))
        act = np.nonzero(b)
        return act

    def copy(self, swap=False):
        return copy.deepcopy(self)

    def swtich_player(self):
        self.curr_player = self.curr_player * -1

    def to_input(self, board=None):
        if board is None:
            board = self.board
        return self.get_layers(board, self.global_wins, self.active_subgame)
