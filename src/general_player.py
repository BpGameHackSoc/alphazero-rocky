from enum import IntEnum
class Player(IntEnum):
    '''
        A player enum of a game.
    '''
    NONE = 0
    ME = 1
    ONE = 1
    OPPONENT = -1
    TWO = -1
    UNKNOWN = 0
    A = 1
    B = -1
    FIRST = 1
    SECOND = -1
    DRAW  = 2