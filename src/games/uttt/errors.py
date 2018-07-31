"""
Errors for the Ultimate TicTacToe Engine
"""

class Error(Exception):
    """Base class for exceptions"""
    pass

class MoveError(Error): pass

class MoveOutsideMainBoardError(MoveError):
    def __init__(self, main_board_coords):
        self.coords = main_board_coords

    def __str__(self):
        return "Co-ordinate ({0}) made outside main board bounds".format(self.coords)


class MoveNotOnNextBoardError(MoveError):
    def __init__(self, main_board_coords, sub_board_next_player_must_play):
        self.main_board_coords = main_board_coords
        self.sub_board_next_player_must_play = sub_board_next_player_must_play

    def __str__(self):
        return "Next board to play is {0}, but player played {1}".format(self.sub_board_next_player_must_play,
                                                                         self.main_board_coords)


class MoveOutsideSubBoardError(MoveError):
    def __init__(self, sub_board_coords):
        self.sub_board_coords = sub_board_coords

    def __str__(self):
        return "Move ({0},{1}) made outside sub board bounds".format(self.sub_board_coords.row,
                                                                     self.sub_board_coords.col)


class MoveInPlayedCellError(MoveError):
    def __init__(self, player, sub_board_coords, main_board_coords=None):
        self.player = player
        self.sub_board_coords = sub_board_coords
        self.main_board_coords = main_board_coords

    def __str__(self):
        msg = "Move {0} made in already played cell by {1}".format(self.sub_board_coords, self.player)
        if self.main_board_coords is not None:
            msg += " in board {0}".format(self.main_board_coords)
        return msg


class MoveInFinishedBoardError(MoveError): pass

class BoardNotFinishedError(MoveError):
    def __str__(self):
        return "Board is currently in play"
