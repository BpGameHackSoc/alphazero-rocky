import numpy as np
from src.arena import Arena
from src.games.gomoku.game import GomokuState
from src.games.gomoku.neural_net import GomokuNN
from src.config import *
import src.games.gomoku.config as gomoku_config
from collections import deque
from copy

class Trainer(object):
    def __init__(self, game_type, model_path=None):
        self.game_type = game_type
        self.model_path = model_path
        self.config()
        self.iterations = NO_OF_ITERATIONS
        self.episodes = NO_OF_EPOSIDES
        self.memory_size = MEMORY_SIZE
        self.observations = deque(maxlen=self.memory_size)

    def config(self):
        if self.game_type == 'gomoku':
            nn = GomokuNN(self.model_path)
            self.best_student = StudentAgent(nn, name='best_student')
            self.start_state = GomokuState()
            self.temp_threshold = gomoku_config.TEMP_THRESHOLD
            self.temp_decay = gomoku_config.TEMP_DECAY

    def play_one_episode(self):
        state = self.start_state
        observations = deque()
        root = None
        while not state.is_over():
            temp = self.__temperature(state.move_count())
            move = self.best_student.move(state, temp=temp, root=root)
            probabilities = self.best_student.last_run['probabilities']
            root = self.student.last_run['chosen_child']
            self.observations.extend(state.to_input(probabilities))
            state = state.move(move)

        # Update value as: 1 for winner, -1 for losers, 0 for draw
        winner = state.winner()
        for i in range(len(observations)):
            observations[i][1] *= winner.value
        return observations

    def learn(self):
        for i in range(self.iterations):
            print(' *** ITERATION : ' + str(i+!) + ' ***')
            for j in range(self.episodes):
                self.observations.extend(self.play_one_episode())
            self.challenger = self.train_counterparty()
            arena = Arena(self.game_type, self.best_student, self.challenger)
            wins = arena.war(NO_OF_GAMES_TO_BATTLE)
            if challanger_takes_crown(wins):
                print('Accepted!')
                self.best_student = challenger
                self.best_student.name = 'best_student'
                self.best_student.nn.save(name=self.game_type+'_checkpoint_'+str(i))
            else:
                print('Rejected!')
        print('Learning has finished.')


    def challanger_takes_crown(self, wins):
        return float(wins[1]) / wins.sum() > SAVE_THRESHOLD


    def train_counterparty(self):
        self.best_student.nn.save('temp', to_print=False)
        challenger_nn = self.__load_appropriate_nn('temp')
        states = [o[0] for o in self.observations]
        values = [o[1] for o in self.observations]
        probabilities = [o[2] for o in self.observations]
        challenger_nn.learn(states, values, probabilities)
        think_time = self.best_student.think_time
        challenger = StudentAgent(challenger_nn, think_time=think_time, name='challenger')
        return challenger


    def __load_appropriate_nn(self, model_name):
        if self.game_type == 'gomoku':
            return GomokuNN(model_name=model_name)

    def __temperature(self, move_count):
        if move_count <= self.temp_threshold:
            return 1
        else:
            return max(0, (1-self.temp_decay*move_count))