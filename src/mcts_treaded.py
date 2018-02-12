import threading
import concurrent.futures
from time import sleep
from src.mcts import MCTS
from src.tree_node import Node_threaded

class MCTS_threaded(MCTS):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)


    def search(self, root_node, simulation_limit=None):
        self.root_node = root_node
        self.root_node.parent = None
        if not root_node.is_visited():
            root_node.evaluate(self.model)
        processes = 4
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=processes) as e:
            for i in range(processes):
                futures.append(e.submit(self.tread_go))
            sleep(self.seconds)  # Time in seconds
            e.shutdown(wait=True)
        return self.root_node

    def tread_go(self):
        while True:
            super().run_one_simulation()

    def simulate_to_leaf(self):
        nodes_visited = []
        node = self.root_node
        while not node.is_terminal():
            with node.update_lock:
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
                node_inner.update_values(v)