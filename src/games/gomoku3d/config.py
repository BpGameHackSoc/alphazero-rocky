# ==============================================================
# GAME
# ==============================================================

# The size of the board on its edge
BOARD_SIZE = 4

# A player needs to connect this many number on the board
CONNECT_SIZE = BOARD_SIZE

# ==============================================================
# NEURAL NET
# ==============================================================

NEURAL_NET_SETTINGS = {
    'no_of_possible_actions' : BOARD_SIZE * BOARD_SIZE,
    'filter_n' : 160,
    'res_layer_n' : 5,
    'input_shape' : (BOARD_SIZE*2, BOARD_SIZE, BOARD_SIZE),
}

# ==============================================================
# TRAINING
# ==============================================================

TEMP_THRESHOLD = BOARD_SIZE
TEMP_DECAY = 1. / 15