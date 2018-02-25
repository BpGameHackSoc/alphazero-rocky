from src.games.gomoku.game import GomokuState
from src.games.gomoku3d.game import Gomoku3dState
from src.games.connect4.game import Connect4State
import numpy as np

class Arena(object):
    def __init__(self, game_type, agent1, agent2):
        self.game_type = game_type
        self.agents = [agent1, agent2]

    def war(self, n, verbose=0):
        '''
            Plays 2*n games between the two agents, n game on both sides.
                verbose: If 0, nothing will be printed.
                         If 1, the game states are printed.
                         If 2, the agents also share their thoughts on the game.
                return: The number of wins on agent1's and agent2 sides respectively.
        '''
        wins = np.array([0., 0.])
        for game_pair_index in range(n):
            wins += self.battle(verbose)
        return wins

    def battle(self, verbose=0):
        '''
            Plays two games between the agents.
                verbose: see at self.war()
        '''
        wins = np.array([0., 0.])
        for starting_agent_index in range(2):
            agent_index = starting_agent_index
            state = self.__init_state()
            while not state.is_over():
                self.__display(verbose, state, self.agents[agent_index])
                move = self.agents[agent_index].move(state)
                state = state.move(move)
                agent_index = 1 - agent_index
            self.__display(verbose, state, self.agents[agent_index])
            wins += self.__determine_scores(state.winner(), starting_agent_index)
        return wins

    def __display(self, verbose, state, agent):
        if verbose >= 1:
            print(state)
        if verbose >= 2:
            agent.evaluate(state)
        if verbose > 0 :
            if state.is_over():
                print('Game over. Winner is: ' + state.winner().str())
            print('=======================================')
            print()

    def __determine_scores(self, winner, starting_agent_index):
        scores = np.array([0., 0.])
        if winner.value == 0:
            scores += 0.5
        elif winner.value == 1:
            scores[starting_agent_index] += 1
        else:
            scores[1-starting_agent_index] += 1
        return scores


    def __init_state(self):
        '''
            Loads in the right game.
        '''
        if self.game_type == 'gomoku':
            return GomokuState()
        if self.game_type == 'gomoku3d':
            return Gomoku3dState()
        if self.game_type == 'connect4':
            return Connect4State()
        raise Exception('Unknown game: ' + str(self.game_type))
