from copy import deepcopy

from .cell import Cell
from .errors import MoveOutsideSubBoardError, MoveInPlayedCellError, \
    MoveInFinishedBoardError, BoardNotFinishedError
from .gameplay import Player, SubBoardCoords
from .gameplay import did_move_win

from src.general_player import Player


class SubBoard(object):
    """A single game of TicTacToe (not ultimate). Several of these make up the Ultimate TTT game.

    Instances of this class behave in a functional manner, with no method call
    modifying the state of the original object. State changing operations (such
    as :code:`add_my_move`) return a new SubBoard object, which the calling function
    must replace. The returned SubBoard object has all properties (e.g. is_finished)
    calculated.

    .. highlight:: python
    Example:
    ::
        SubBoard(3) #Initialises a board of size 3
            .add_my_move(SubBoardCoords(1, 1)) #Adds a move at 1, 1 and returns a SubBoard
            .add_opponent_move(SubBoardCoords(2, 1)) #Adds a move to the last returned board

    Call :code:`str(SubBoard())` to get a pretty-printed representation of this board

    Todo:
        * Use a :code:`@classmethod` to initialize SubBoard and make it immutable internally

    """

    def __init__(self, board_size: int = 3):
        if not board_size == 3:
            raise ValueError("Size must be integer of size 3 (for now)")

        self.board_size = board_size
        self._board = [
            [Cell() for board_col in range(board_size)]
            for board_row in range(board_size)
        ]

        self.max_moves = board_size * board_size
        self.moves_so_far = 0

        self._is_finished = False
        self._winner = Player.NONE

    @property
    def is_finished(self) -> bool:
        """Whether the board is finished (tied, won or lost)"""
        return self._is_finished

    @property
    def winner(self) -> Player:
        """The winner of the board if finished. Exception otherwise"""
        if not self._is_finished:
            raise BoardNotFinishedError
        return self._winner

    def add_my_move(self, sub_board_coords: SubBoardCoords) -> 'SubBoard':
        """Adds a move for the current player

        Args:
            sub_board_coords: Move co-ordinates

        Returns:
            A new SubBoard instance with the move applied
        """
        return self.add_move(sub_board_coords, Player.ME)

    def add_opponent_move(self, sub_board_coords: SubBoardCoords) -> 'SubBoard':
        """Adds a move for the opponent

        Args:
            sub_board_coords: Move co-ordinates

        Returns:
            A new SubBoard instance with the move applied
        """
        return self.add_move(sub_board_coords, Player.OPPONENT)

    def __str__(self):
        """Returns a pretty printed representation of this board"""
        pretty_printed = ''
        for row in self._board:
            for cell in row:
                pretty_printed += str(cell) + ' '
            pretty_printed += '\n'
        return pretty_printed

    def add_move(self, sub_board_coords: SubBoardCoords, player: Player) -> 'SubBoard':
        """Adds a move by a ultimate_ttt_player to a deep copy of the board, returning the copy

        Player may find it easier to use the :func:`~add_my_move` and
        :func:`~add_opponent_move` functions

        Args:
            sub_board_coords: The co-ordinates to make a move in on this sub-board
            player: The player that made the move

        Returns:
            A new SubBoard instance with the move applied and all properties calculated
        """
        if self.is_finished:
            raise MoveInFinishedBoardError(sub_board_coords, player)

        if not (self._is_move_in_bounds(sub_board_coords)):
            raise MoveOutsideSubBoardError(sub_board_coords)

        if self._is_move_already_played(sub_board_coords):
            raise MoveInPlayedCellError(player, sub_board_coords)

        # Copy the board so we can update it
        # Maybe this should all go in the constructor/classmethod
        updated_sub_board = deepcopy(self)

        updated_sub_board._board[sub_board_coords.row][sub_board_coords.col] = Cell(player)
        updated_sub_board.moves_so_far += 1

        if did_move_win(updated_sub_board._board, sub_board_coords, player):
            updated_sub_board._is_finished = True
            updated_sub_board._winner = player
        elif updated_sub_board.moves_so_far == updated_sub_board.max_moves:
            updated_sub_board._is_finished = True

        return updated_sub_board

    def get_playable_coords(self) -> [SubBoardCoords]:
        """
        Returns:
            All valid SubBoardCoords that can be played (have not been played

            Empty if board is finished.
        """
        if self.is_finished:
            return []
        valid_coords = []
        for row_index in range(0, self.board_size):
            for col_index in range(0, self.board_size):
                if not self._board[row_index][col_index].is_played():
                    valid_coords.append(SubBoardCoords(row_index, col_index))
        return valid_coords

    def to_input_generator(self):
        # (np.float(cell.played_by) for row in mb.board[0][0] for cell in row)
        pass


    # Private functions
    def _is_move_in_bounds(self, sub_board_coords: SubBoardCoords) -> bool:
        """Checks whether the given move is inside the boundaries of this board

        Args:
            sub_board_coords: The intended move

        Returns:
            True if the move is within the bounds of this board, False otherwise
        """
        if 0 <= sub_board_coords.row < len(self._board) and 0 <= sub_board_coords.col < len(self._board):
            return True
        return False

    def _is_move_already_played(self, sub_board_coords: SubBoardCoords) -> bool:
        """Checks whether the given move is already played in this board

        Args:
            sub_board_coords: The intended move

        Returns:
            True if the cell referenced by the move is already played, False otherwise
        """
        return self._board[sub_board_coords.row][sub_board_coords.col].is_played()

    def __iter__(self):
        return self._board.__iter__()

    def __getitem__(self, key):
        return self._board[key]
