import numpy as np
from src.arena import Arena
from src.games.gomoku.game import GomokuState
from src.games.gomoku.neural_net import GomokuNN
from src.config import *
import src.games.gomoku.config as gomoku_config
from collections import deque

import src.agents.student
from importlib import reload
reload(src.agents.student)
from src.agents.student import StudentAgent
from copy import deepcopy

parallelize_episodes=True
if parallelize_episodes:
    threads = 8
    import concurrent.futures

class Trainer(object):
    def __init__(self, game_type, model_path=None):
        self.game_type = game_type
        self.model_path = model_path  
        self.iterations = NO_OF_ITERATIONS
        self.episodes = NO_OF_EPISODES
        self.memory_size = MEMORY_SIZE
        self.observations = deque(maxlen=self.memory_size)
        self.config()

    def config(self):
        if self.game_type == 'gomoku':
            nn = GomokuNN(self.model_path)
            self.best_student = StudentAgent(nn, name='best_student')
            self.start_state = GomokuState()
            self.temp_threshold = gomoku_config.TEMP_THRESHOLD
            self.temp_decay = gomoku_config.TEMP_DECAY

    @staticmethod
    def play_one_episode(student,start_state,temp_threshold,temp_decay):
        state = start_state.copy()
        observations = deque()
        root = None
        while not state.is_over():
            temp = Trainer.temperature(state.move_count(),temp_threshold,temp_decay)
            move = student.move(state, temp=temp, root=root)
            probabilities = student.last_run['probabilities']
            root = student.last_run['chosen_child']
            observations.extend(state.to_all_symmetry_input(probabilities))
            state = state.move(move)

        # Update value as: 1 for winner, -1 for losers, 0 for draw
        winner = state.winner()
        for i in range(len(observations)):
            observations[i][1] *= winner.value
        return observations

    @staticmethod
    def process_play_episodes(num_of_episodes, nn_path, think_time,clone_name,start_state,temp_threshold,temp_decay):
        clone_nn = GomokuNN(nn_path)
        student =StudentAgent(clone_nn,think_time=think_time,name=clone_name)
        observations = []
        for i in range(num_of_episodes):
            observations.extend(Trainer.play_one_episode(student,start_state,temp_threshold,temp_decay))
        return observations


    def learn(self):
        for i in range(self.iterations):
            print(' *** ITERATION : ' + str(i+1) + ' ***')
            if parallelize_episodes:
                self.best_student.nn.save('clone_nn')
                with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as e:
                    futures = []
                    min_work = int(self.episodes/threads) #divide work between processes evenly
                    work_boundary = self.episodes % threads
                    for i in range(threads):
                        work = min_work
                        if i<work_boundary:
                            work+=1
                        st = deepcopy(self.start_state)
                        tt = deepcopy(self.temp_threshold)
                        td = deepcopy(self.temp_decay)
                        futures.append(e.submit(Trainer.process_play_episodes,work,'clone_nn',self.best_student.name,self.best_student.think_time,st,tt,td))
                    done_futures,_ = concurrent.futures.wait(futures) #wait until processes finish
                    for df in done_futures: #loop through the results
                        self.observations.extend(df.result())
            else:
                for j in range(self.episodes):
                    self.observations.extend(self.play_one_episode())
            self.challenger = self.train_counterparty()
            arena = Arena(self.game_type, self.best_student, self.challenger)
            wins = arena.war(NO_OF_GAMES_TO_BATTLE)
            if self.challanger_takes_crown(wins):
                print('Accepted!')
                self.best_student = self.challenger
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

    @staticmethod
    def temperature(move_count,temp_threshold, temp_decay):
        if move_count <= temp_threshold:
            return 1
        else:
            return max(0, (1-temp_decay*move_count))