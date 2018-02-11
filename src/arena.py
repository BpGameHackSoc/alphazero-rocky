from src.games.gomoku.game import GomokuState
import numpy as np

class Arena(object):
    def __init__(self, game_type, player1, player2):
        self.game_type = game_type
        self.players = [player1, player2]

    def war(self, n, verbose=0):
        '''
            Plays 2*n games between the two agents, n game on both sides.
                verbose: If 0, nothing will be printed.
                         If 1, the game states are printed.
                         If 2, the agents also share their thoughts on the game.
                return: The number of wins on player1's and player2 sides respectively.
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
        for starting_player_index in range(2):
            player_index = starting_player_index
            state = self.__init_state()
            while not state.is_over():
                self.__display(verbose, state, self.players[player_index])
                move = self.players[player_index].move(state)
                state = state.move(move)
                player_index = 1 - player_index
            self.__display(verbose, state, self.players[player_index])
            wins += self.__determine_scores(state.winner(), starting_player_index)
        return wins

    def __display(self, verbose, state, player):
        if verbose >= 1:
            print(state)
        if verbose >= 2:
            player.evaluate(state)
        if verbose > 0 :
            if state.is_over():
                print('Game over. Winner is: ' + state.winner().str())
            print('=======================================')
            print()

    def __determine_scores(self, winner, starting_player_index):
        scores = np.array([0., 0.])
        if winner.value == 0:
            scores += 0.5
        elif winner.value == 1:
            scores[starting_player_index] += 1
        else:
            scores[1-starting_player_index] += 1
        return scores


    def __init_state(self):
        '''
            Loads in the right game.
        '''
        if self.game_type == 'gomoku':
            return GomokuState()
        raise Exception('Unknown game: ' + str(self.game_type))
