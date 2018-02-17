import numpy as np
from src.arena import Arena
from src.games.gomoku.game import GomokuState
from src.games.gomoku.neural_net import GomokuNN
from src.config import *
import src.games.gomoku.config as gomoku_config
from collections import deque
import pickle

import src.agents.student
from importlib import reload
reload(src.agents.student)
from src.agents.student import StudentAgent

class Trainer(object):
    def __init__(self, game_type, model_path=None, memory_path=None):
        self.game_type = game_type
        self.model_path = model_path  
        self.iterations = NO_OF_ITERATIONS
        self.episodes = NO_OF_EPISODES
        self.memory_size = MEMORY_SIZE
        self.no_of_games_played = self.__get_no_of_games_played(model_path)
        self.observations = self.__get_initial_observations(memory_path)
        self.config()

    def config(self):
        if self.game_type == 'gomoku':
            nn = GomokuNN(self.model_path)
            self.best_student = StudentAgent(nn, name='best_student')
            self.start_state = GomokuState()
            self.temp_threshold = gomoku_config.TEMP_THRESHOLD
            self.temp_decay = gomoku_config.TEMP_DECAY

    def play_one_episode(self):
        state = self.start_state.copy()
        observations = deque()
        root = None
        possible_move_len = state.action_space_size()
        while not state.is_over():
            temp = self.__temperature(state.move_count())
            move = self.best_student.move(state, temp=temp, root=root)
            probabilities = self.best_student.last_run['probabilities']
            root = self.best_student.last_run['chosen_child']
            observations.extend(state.to_all_symmetry_input(probabilities))
            state = state.move(move)

        # Saving last state in order determine what means winning
        probabilities = np.full((possible_move_len, ), 1./possible_move_len)
        observations.extend(state.to_all_symmetry_input(probabilities))

        # Update value as: 1 for winner, -1 for losers, 0 for draw
        winner = state.winner()
        for i in range(len(observations)):
            observations[i][1] *= winner.value
        return observations

    def learn(self):
        n = self.no_of_games_played
        for i in range(n, n + self.iterations):
            print(' *** ITERATION : ' + str(i+1) + ' ***')
            for j in range(self.episodes):
                self.observations.extend(self.play_one_episode())
            self.challenger = self.train_counterparty()
            self.best_student.learning = False
            self.challenger.learning = False
            arena = Arena(self.game_type, self.best_student, self.challenger)
            wins = arena.war(NO_OF_GAMES_TO_BATTLE)
            if self.challanger_takes_crown(wins):
                print('Accepted!')
                self.best_student = self.challenger
                self.best_student.name = 'best_student' 
                self.best_student.nn.save(file_name=self.game_type+'_checkpoint_'+str(i+1))
            else:
                print('Rejected!')
            self.best_student.learning = True
        print('Learning has finished.')
        self.save_memory = self.__save_memory('memory_' + str(n+self.iterations))


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

    def __save_memory(self, file_name):
        path = WORK_FOLDER + file_name + '.p'
        pickle.dump(self.observations, open(path, "wb" ))
        print('Memory saved at ' + path)


    def __get_no_of_games_played(self, s):
        if s == None:
            return 0
        tofind = 'checkpoint_'
        loc = s.find('checkpoint_')
        if loc == -1:
            return 0
        else:
            return int(s[len(tofind)+loc:])
        if self.game_type == 'gomoku':
            return GomokuNN(model_name=model_name)

    def __get_initial_observations(self, memory_name):
        observations = deque(maxlen=self.memory_size)
        if memory_name is None:
            return observations
        path = WORK_FOLDER + memory_name + '.p'
        old_observations = pickle.load(open(path, "rb" ))
        observations.extend(old_observations)
        return observations

    def __load_appropriate_nn(self, model_name):
        if self.game_type == 'gomoku':
            return GomokuNN(model_name=model_name)

    def __temperature(self, move_count):
        if move_count <= self.temp_threshold:
            return 1
        else:
            return max(0, (1-self.temp_decay*move_count))