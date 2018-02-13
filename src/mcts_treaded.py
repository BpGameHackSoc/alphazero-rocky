import threading
import concurrent.futures
from time import sleep
from src.mcts import MCTS
from src.tree_node import Node_threaded


done = False
def tread_go(cls):
    while not cls.done:
        cls.run_one_simulation()

class MCTS_threaded(MCTS):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)


    def search(self, root_node, threads = 1,simulation_limit=None):
        self.root_node = root_node
        self.root_node.parent = None
        if not root_node.is_visited():
            root_node.evaluate(self.model)
        futures = []
        self.done = False
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as e:
            for i in range(threads):
                futures.append(e.submit(tread_go,self))
            sleep(self.seconds)  # Time in seconds
            self.done = True
        return self.root_node


    def simulate_to_leaf(self):
        nodes_visited = []
        node = self.root_node
        while not node.is_terminal():
            with node.update_lock:
                node.update_values(-1) #virtual loss
                nodes_visited.append(node)
                move_index = self.select(node)
                new_node = node.children[move_index]
                if new_node is None:
                    node = node.expand_and_evaluate(move_index,self.model)
                    break
                node = new_node
        return (node,nodes_visited)

    def backpropagation(self, node,nodes_visited):
        with node.update_lock:
            v = node.get_V()
        for node_inner in nodes_visited:
            with node_inner.update_lock:
                MCTS_threaded.update_and_rollback_vl(node_inner,v)
    @staticmethod
    def update_and_rollback_vl(node, estimate):
        node.W += 1 + estimate
        node.calc_Q()
