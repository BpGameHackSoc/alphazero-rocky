import unittest
import numpy as np

import sys
sys.path.append('..')
from ..game import UTTTState
from src.arena import Arena
from src.agents.random import RandomAgent


class UTTTStateTester(unittest.TestCase):

    def setUp(self):
        self.state = UTTTState()

    def test_first_moves(self):
        valid_moves = [(0, 0, 0, 0),
                       (2, 2, 0, 0),
                       (2, 2, 2, 2),
                       (1, 2, 0, 2),
                       (1, 1, 1, 1)]
        for move in valid_moves:
            _ = self.state.move(move)
        self.assertTrue(True)

    def test_valid_moves(self):
        valid_moves = self.state.valid_moves()
        expected = sorted(np.array(np.meshgrid(*[range(3)] * 4)).T.reshape(-1, 4).tolist())
        self.assertEqual(expected, sorted(valid_moves.tolist()))

    def test_valid_first_moves(self):
        valid_moves = self.state.valid_moves()
        for move in valid_moves:
            _ = self.state.move(move)
        self.assertTrue(True)

    def test_random_agents(self):
        r = RandomAgent()
        a = Arena('uttt', r, r)
        wins, history = a.war(100)
        self.assertTrue(True)


    # def test_first_moves(self):
    #     valid_moves = [(0, 0, 0, 0),
    #                    (2, 2, 0, 0),
    #                    (2, 2, 2, 2),
    #                    (1, 2, 0, 2),
    #                    (1, 1, 1, 1)]
    #     for move in valid_moves:
    #         _ = self.state.move(move)

if __name__ == '__main__':
    unittest.main()