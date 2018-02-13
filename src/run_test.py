from src.games.gomoku.game import GomokuState
from src.games.gomoku.neural_net import GomokuNN
from src  import mcts
from src  import mcts_treaded
from src  import tree_node
from src.games.gomoku.config import BOARD_SIZE
# Get info about how fast the MCTS is
b = GomokuNN()
g = GomokuState()
time = 2
threads = True
if threads:
    for i in range(1,100,20):
        n = tree_node.Node_threaded(g)
        ts = mcts_treaded.MCTS_threaded(b, time=time)
        root = ts.search(n,threads=i)
        print(ts.stats())
else:
    n = tree_node.Node(g, BOARD_SIZE*BOARD_SIZE)
    ts = mcts.MCTS(b, time=time)
    root = ts.search(n)
    print(ts.stats())
pass
