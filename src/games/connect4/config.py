# ==============================================================
# GAME
# ==============================================================

# The size of the board on its edge
BOARD_SIZE = 7

# A player needs to connect this many number on the board
CONNECT_SIZE = 4

# ==============================================================
# NEURAL NET
# ==============================================================

NEURAL_NET_SETTINGS = {
    'no_of_possible_actions' : BOARD_SIZE,
    'filter_n' : 32,
    'res_layer_n' : 2,
    'input_shape' : (2,BOARD_SIZE,BOARD_SIZE),
}

# ==============================================================
# TRAINING
# ==============================================================

LOW_TEMP_THRESHOLD = CONNECT_SIZE
HIGH_TEMP_THRESHOLD = BOARD_SIZE-1
TEMP_DECAY = 1. / BOARD_SIZE