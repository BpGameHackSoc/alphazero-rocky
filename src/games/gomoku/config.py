# ==============================================================
# GAME
# ==============================================================

# The size of the board on its edge
BOARD_SIZE = 3

# A player needs to connect this many number on the board
CONNECT_SIZE = 3

# ==============================================================
# NEURAL NET
# ==============================================================

NEURAL_NET_SETTINGS = {
    'no_of_possible_actions' : BOARD_SIZE * BOARD_SIZE,
    'filter_n' : 64,
    'res_layer_n' : 2,
    'input_shape' : (2,3,3),
}

# ==============================================================
# TRAINING
# ==============================================================

TEMP_THRESHOLD = BOARD_SIZE
TEMP_DECAY = 1. / BOARD_SIZE