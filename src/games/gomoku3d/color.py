from enum import IntEnum
class Color(IntEnum):
    '''
        The Color class represents a side to move or the winner.
    '''
    NONE = 0
    BLUE = 1
    RED = -1

    def str(self):
        if self.value == 0:
            return '-'
        elif self.value == 1:
            return 'X'
        else:
            return 'O'