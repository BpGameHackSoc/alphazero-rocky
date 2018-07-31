from copy import deepcopy
import numpy as np

from .sub_board import SubBoard
from .cell import Cell
from .gameplay import Player, MainBoardCoords, SubBoardCoords,Move
from .gameplay import did_move_win
from .errors import MoveOutsideMainBoardError, MoveNotOnNextBoardError, \
    BoardNotFinishedError, MoveInFinishedBoardError, \
    MoveInPlayedCellError

from src.general_game import GameState
from src.general_player import Player


class MainBoard(GameState):
    """An Ultimate TicTacToe board, containing several SubBoards where players play

    When the board size is 3, the main board looks like this:

    ::

        | SubBoard 0,0 | SubBoard 0,1 | SubBoard 0,2 |
        | SubBoard 1,0 | SubBoard 1,1 | SubBoard 1,2 |
        | SubBoard 2,0 | SubBoard 2,1 | SubBoard 2,2 |

    Each SubBoard looks like this:

    ::

        | Cell 0,0 | Cell 0,1 | Cell 0,2 |
        | Cell 1,0 | Cell 1,1 | Cell 1,2 |
        | Cell 2,0 | Cell 2,1 | Cell 2,2 |

    """

    def __init__(self, board_size: int = 3):
        if not board_size == 3:
            raise ValueError("Size must be 3 (for now)")

        self._board_size = board_size
        self.board = [
            [SubBoard() for board_col in range(board_size)]
            for board_row in range(board_size)
        ]

        self._next_player = Player.NONE  # type: Player
        self._sub_board_next_player_must_play = None  # type: MainBoardCoords

        self._is_finished = False  # type: bool
        self._winner = Player.NONE  # type: Player

    @property
    def sub_board_next_player_must_play(self) -> MainBoardCoords:
        """The next board to play on. None if the next move can be on any board"""
        return self._sub_board_next_player_must_play

    @property
    def is_finished(self) -> bool:
        """Whether the board is finished (tied, won or lost)"""
        if self._is_finished:
            return True

        for row in self.board:
            for sub_board in row:
                if not sub_board.is_finished:
                    return False

        return True

    @property
    def winner(self) -> Player:
        """The winner of the board if finished. Exception otherwise"""
        if not self.is_finished:
            raise BoardNotFinishedError
        return self._winner

    def add_my_move(self, move_obj) -> 'MainBoard':
        """Adds your move to the specified sub-board

        Args:
            main_board_coords: The co-ordinates (row, column) of the SubBoard to play on
            move: The move (row, column) to make on the SubBoard

        Returns:
            A new MainBoard instance with the move applied
        """
        return self.move(move_obj, Player.ME)

    def add_opponent_move(self, move_obj) -> 'MainBoard':
        """Adds the opponent's move to the specified sub-board

        Args:
            main_board_coords: The co-ordinates (row, column) of the SubBoard to play on
            sub_board_coords: The move (row, column) to make on the SubBoard

        Returns:
            A new MainBoard instance with the move applied
        """
        return self.move(move_obj, Player.OPPONENT)

    def get_sub_board(self, main_board_coords: MainBoardCoords) -> SubBoard:
        row = main_board_coords.row
        col = main_board_coords.col
        return self.board[row][col]

    def get_playable_coords(self) -> [MainBoardCoords]:
        """Returns all board co-ordinates that are valid for the next move.

        If the opponents previous move co-ordinates (according to the rules)
        restrict you to a single sub-board, then this will return only that board.
        If not, it will return all boards that are valid for moves. 

        Returns Empty if board is finished.

        Returns:
            Array of valid board co-ordinates (Row, Col), e.g. [MainBoardCoords(2, 2),MainBoardCoords(1, 1)]
        """
        if self.sub_board_next_player_must_play is not None:
            return [self.sub_board_next_player_must_play]
        else:
            if self.is_finished:
                return []
            available_boards = []
            for row_index in range(0, self._board_size):
                for col_index in range(0, self._board_size):
                    if not self.board[row_index][col_index].is_finished:
                        available_boards.append(MainBoardCoords(row_index, col_index))
            return available_boards

    def is_playing_on_sub_board_allowed(self, main_board_coords: MainBoardCoords):
        """Whether this is a valid board for the next move

        Args:
            main_board_coords: The co-ordinates (row, column) of the SubBoard to check
        """
        if self.sub_board_next_player_must_play is None or \
                self.sub_board_next_player_must_play == main_board_coords:
            return True
        return False

    def __str__(self):
        """Returns a pretty printed representation of the main board"""
        pretty_printed = ''
        # TODO: Shouldn't access sub-board private var
        board_size = len(self.board)
        board_size_range = range(board_size)
        for (mb_idx, mb_row) in enumerate(self.board):
            for sub_board_row_num in board_size_range:
                for (sb_idx, sub_board) in enumerate(mb_row):
                    for cell in sub_board._board[sub_board_row_num]:
                        pretty_printed += str(cell) + ' '
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

    # Private functions
    def move(self, move_obj, player) -> 'MainBoard':
        """Adds a move by a ultimate_ttt_player to a deep copy of the current board, returning the copy

        Args:
            main_board_coords: The location of the sub-board on the main board

        Returns:
            A new MainBoard instance with the move applied and all properties calculated
        """
        main_board_coords  = move_obj.main
        if self._is_finished:
            raise MoveInFinishedBoardError(main_board_coords, player)

        if not self._is_board_in_bounds(main_board_coords):
            raise MoveOutsideMainBoardError(main_board_coords)

        if not self.is_playing_on_sub_board_allowed(main_board_coords):
            raise MoveNotOnNextBoardError(main_board_coords, self._sub_board_next_player_must_play)

        return self.copy_applying_move(move_obj, player)

    def copy_applying_move(self, move_obj, player) -> 'MainBoard':
        main_board_coords = move_obj.main
        move_obj = move_obj.sub
        # Apply the move to the sub board first to ensure it works
        try:
            updated_sub_board = self.board[main_board_coords.row][main_board_coords.col] \
                .add_move(move_obj, player)
        except MoveInPlayedCellError as e:
            raise MoveInPlayedCellError(player, move_obj, main_board_coords) from e

        # Copy the board so we can update it
        # Maybe this should all go in the constructor/classmethod
        updated_main_board = deepcopy(self)

        updated_main_board.board[main_board_coords.row][main_board_coords.col] = updated_sub_board

        # Check that the next board to play is not finished
        if not updated_main_board.board[move_obj.row][move_obj.col].is_finished:
            updated_main_board._sub_board_next_player_must_play = MainBoardCoords(move_obj.row,
                                                                                  move_obj.col)
        else:
            updated_main_board._sub_board_next_player_must_play = None

        # Convert to board of cells format so we can reuse check logic
        cell_board = updated_main_board._as_cell_board()

        if did_move_win(cell_board, main_board_coords, player):
            updated_main_board._is_finished = True
            updated_main_board._winner = player

        return updated_main_board

    def _as_cell_board(self) -> [[Cell]]:
        """Returns this main board in the form of a board of cells

        Each cell represents a sub-board of this board, with
        `cell.played_by` set to the ultimate_ttt_player that won the board (Player.NONE if tied)
        """

        return [[Cell(sub_board.winner) if sub_board.is_finished else Cell(Player.NONE) for sub_board in row] for row in
                self.board]

    def _is_board_in_bounds(self, coords) -> bool:
        """Checks whether the given move is inside the boundaries of the main board

        Args:
            coords: The coords of the intended sub-board

        Returns:
            True if the move is within the bounds of the main board, False otherwise
        """

        if 0 <= coords.row < len(self.board) and 0 <= coords.col < len(self.board):
            return True
        return False

    def turn(self):
        return self._next_player

    def move_count(self):
        return sum(b.moves_so_far for row in self.board for b in row)

    def valid_moves(self):
        mainboards = self.get_playable_coords()
        moves = [Move(main,self.get_sub_board(main)) for main in mainboards]
        return moves

    def action_space_size(self):
        sub_board_size = self.board[0].board_size
        return self._board_size*self._board_size*sub_board_size*sub_board_size

    def is_over(self):
        return self.is_finished

    def to_input(self):
        pass

    def copy(self):
        pass

    def to_all_symmetry_input(self):
        pass

    def __iter__(self):
        return self.board.__iter__()

    def __getitem__(self, key):
        return self.board[key]
