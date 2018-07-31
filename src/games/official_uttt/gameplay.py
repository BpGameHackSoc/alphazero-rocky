from .cell import Cell
from .player import Player
from src.general_player import Player

class Move:
    """Holds all information about a move that transfers the game from state to state.
    Holds a MainBoardCoords and a SubBoardCoords as members"""
    def __init__(self, main, sub):
        self.main = main
        self.sub = sub

class SubBoardCoords(object):
    """Move co-ordinates (in a SubBoard)"""

    def __init__(self, row, col):
        self.row = row
        self.col = col

    def __eq__(self, other):
        return not (other == None) and \
               self.row == other.row and self.col == other.col

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return "(" + str(self.row) + "," + str(self.col) + ")"


class MainBoardCoords(SubBoardCoords):
    """Convenience wrapper to represent MainBoard co-ordinates (referencing a SubBoard)"""

    def __init__(self, main_board_row, main_board_col):
        super().__init__(main_board_row, main_board_col)


def did_move_win(board: [[Cell]], board_coords: SubBoardCoords, player: Player) -> bool:
    """Whether the given player move was a winning move (this assumes that the move is already present in the board)"""
    return (is_row_won(board, board_coords, player) or is_col_won(board, board_coords, player) or
            is_diagonal_won(board, player))


def is_row_won(board: [[Cell]], board_coords: SubBoardCoords, player: Player) -> bool:
    """Whether the row of the player move is won by the player of the move"""
    return is_cell_range_played_by(board[board_coords.row], player)


def is_col_won(board: [[Cell]], board_coords: SubBoardCoords, player: Player) -> bool:
    """Whether the column of the player move is won by the player of the move"""
    for row in board:
        if not row[board_coords.col].played_by == player:
            return False
    return True


def is_diagonal_won(board: [[Cell]], player: Player) -> bool:
    """Whether either diagonal from the cell of the player move is won by the player"""

    return is_ltr_diagonal_won(board, player) or is_rtl_diagonal_won(board, player)


def is_ltr_diagonal_won(board: [[Cell]], player: Player) -> bool:
    """Whether the left to right (0,0) to (2,2) diagonal has been won by the given player"""
    cells = [board[0][0], board[1][1], board[2][2]]

    return is_cell_range_played_by(cells, player)


def is_rtl_diagonal_won(board: [[Cell]], player: Player) -> bool:
    """Whether the left to right (2,0) to (0,2) diagonal has been won by the given player"""
    cells = [board[2][0], board[1][1], board[0][2]]

    return is_cell_range_played_by(cells, player)


def is_cell_range_played_by(cells: [[Cell]], player: Player) -> bool:
    """Whether the given list of cells are all played by the given player

    Args:
        cells: The list of cells to check
        player: The player to look for
    """
    if any(not cell.played_by == player for cell in cells):
        return False
    return True
