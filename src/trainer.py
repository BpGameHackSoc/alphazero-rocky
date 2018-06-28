import numpy as np
from src.arena import Arena
from src.games.gomoku.game import GomokuState
from src.games.gomoku3d.game import Gomoku3dState
from src.games.connect4.game import Connect4State
from src.games.gomoku.neural_net import GomokuNN
from src.games.gomoku3d.neural_net import Gomoku3dNN
from src.games.connect4.neural_net import Connect4NN
from src.config import *
import src.games.gomoku.config as gomoku_config
import src.games.gomoku3d.config as gomoku3d_config
import src.games.connect4.config as connect4_config
from collections import deque
import pickle
from tqdm import tqdm, trange

import src.agents.student
from importlib import reload
reload(src.agents.student)
from src.agents.student import StudentAgent
from src.agents.random import RandomAgent
from time import sleep

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
            self.start_state = GomokuState()
            self.low_temp_threshold = gomoku_config.LOW_TEMP_THRESHOLD
            self.high_temp_threshold = gomoku_config.HIGH_TEMP_THRESHOLD
            self.temp_decay = gomoku_config.TEMP_DECAY
        if self.game_type == 'gomoku3d':
            nn = Gomoku3dNN(self.model_path)
            self.start_state = Gomoku3dState()
            self.low_temp_threshold = gomoku3d_config.LOW_TEMP_THRESHOLD
            self.high_temp_threshold = gomoku3d_config.HIGH_TEMP_THRESHOLD
            self.temp_decay = gomoku3d_config.TEMP_DECAY
        if self.game_type == 'connect4':
            nn = Connect4NN(self.model_path)
            self.start_state = Connect4State()
            self.low_temp_threshold = connect4_config.LOW_TEMP_THRESHOLD
            self.high_temp_threshold = connect4_config.HIGH_TEMP_THRESHOLD
            self.temp_decay = connect4_config.TEMP_DECAY
        self.best_student = StudentAgent(nn, name='best_student')

    def fill_memory_with_random_plays(self):
        agent = RandomAgent()
        possible_move_len = self.start_state.action_space_size()
        p = np.full((possible_move_len, ), 1./possible_move_len)
        pbar = tqdm(total=self.memory_size, desc='Random fill')
        while len(self.observations) < self.memory_size:
            current_observations = deque()
            state = self.start_state.copy()
            while not state.is_over():
                noise = np.random.uniform(-0.1/p.size, 0.1/p.size, p.size)
                noise -= noise.mean()
                move = agent.move(state)
                current_observations.extend(state.to_all_symmetry_input(p+noise))
                state = state.move(move)
            noise = np.random.uniform(-0.1/p.size, 0.1/p.size, p.size)
            noise -= noise.mean()
            current_observations.extend(state.to_all_symmetry_input(p+noise))
            winner = state.winner()
            for i in range(len(current_observations)):
                current_observations[i][1] *= winner.value
            self.observations.extend(current_observations)
            pbar.update(len(current_observations))
            sleep(0.01)
        pbar.close()

    def play_one_episode(self):
        state = self.start_state.copy()
        observations = deque()
        rewards = deque()
        obs_extend = observations.extend
        rew_append = rewards.append
        root = None
        possible_move_len = state.action_space_size()
        while not state.is_over():
            temp = self.__temperature(state.move_count())
            move = self.best_student.move(state, temp=temp, root=root)
            probabilities = self.best_student.last_run['probabilities']
            root = self.best_student.last_run['chosen_child']
            obs_extend(state.to_all_symmetry_input(probabilities))
            rew_append([self.best_student.last_run['predicted_outcome'],
                        self.best_student.last_run['confidence']])
            state = state.move(move)

        # Saving last state in order determine what means winning
        probabilities = self.best_student.last_run['probabilities']
        probabilities = np.full((possible_move_len, ), 1./possible_move_len)
        obs_extend(state.to_all_symmetry_input(probabilities))

        # Update value as: 1 for winner, -1 for losers, 0 for draw
        winner = state.winner()
        for i in range(len(observations)):
            observations[i][1] *= winner.value
        return observations

    def learn(self):
        n = self.no_of_games_played
        for i in range(n, n + self.iterations):
            sleep(0.1)
            tqdm.write(' *** ITERATION : ' + str(i+1) + ' ***')
            # if len(self.observations) == 0:
            #     tqdm.write(' - Using random plays, memory is empty.. - ')
            #     sleep(0.3)
            #     self.fill_memory_with_random_plays()
            # else:
            sleep(0.1)
            for j in trange(self.episodes):
                self.observations.extend(self.play_one_episode())
            self.challenger = self.train_counterparty()
            self.best_student.learning = False
            self.challenger.learning = False
            arena = Arena(self.game_type, self.best_student, self.challenger)
            wins = arena.war(NO_OF_GAMES_TO_BATTLE)
            sleep(0.1)
            tqdm.write('Match result is : ' +  str(wins))
            if self.challanger_takes_crown(wins):
                tqdm.write('Accepted!')
                self.best_student = self.challenger
                self.best_student.name = 'best_student' 
                self.best_student.nn.save(file_name=self.game_type+'new_checkpoint_'+str(i+1))
            else:
                tqdm.write('Rejected!')
            self.best_student.learning = True
        tqdm.write('Learning has finished.')
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
        path = WORK_FOLDER + self.game_type + '_' + file_name + '.p'
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
        if self.game_type == 'gomoku3d':
            return Gomoku3dNN(model_name=model_name)

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
        if self.game_type == 'gomoku3d':
            return Gomoku3dNN(model_name=model_name)
        if self.game_type == 'connect4':
            return Connect4NN(model_name=model_name)

    def __temperature(self, move_count):
        if move_count <= self.low_temp_threshold:
            return 2
        if move_count <= self.high_temp_threshold:
            return 1
        else:
            return max(0, (1-self.temp_decay*move_count))