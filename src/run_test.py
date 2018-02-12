from src.test_teacher import Teacher,Gomoku,Brain
from src.games.gomoku.game import GomokuState
from src.games.gomoku.neural_net import GomokuNN
from src  import mcts
from src  import tree_node
from src.games.gomoku.config import BOARD_SIZE
# Get info about how fast the MCTS is
b = GomokuNN()
g = GomokuState()
n = tree_node.Node(g, BOARD_SIZE*BOARD_SIZE)
ts = mcts.MCTS(b, time=3)
root = ts.search(n,3)
pass