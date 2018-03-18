import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Adds project dir to PYTHONPATH
from src.games.gomoku3d.game import Gomoku3dState
from src.games.gomoku3d.neural_net import Gomoku3dNN
from src  import mcts
from src  import mcts_treaded
from src  import tree_node
import profiler
from contextlib import redirect_stdout
from src.games.gomoku.config import BOARD_SIZE
# Get info about how fast the MCTS is
b = Gomoku3dNN()
b.load('_',path='/home/gergely/Workspace/rocky/bin/gomoku3dnew_checkpoint_71.h5')
g = Gomoku3dState()
time = 2
threads = False
with open('Cpython.txt', 'w') as f:
    with redirect_stdout(f):
        with profiler.Profiler():
            if threads:
                for i in range(1,100,20):
                    n = tree_node.Node_threaded(g)
                    ts = mcts_treaded.MCTS_threaded(b, time=time)
                    root = ts.search(n,threads=i)
                    print(ts.stats())
            else:
                n = tree_node.Node(g)
                ts = mcts.MCTS(b, time=time)
                root = ts.search(root=n)
                print(ts.stats())
pass
