from src.general_nn import NeuralNetwork
from src.games.gomoku.config import NEURAL_NET_SETTINGS

class GomokuNN(NeuralNetwork):
    def __init__(self, model_name=None):
        self.config()
        for key, value in NEURAL_NET_SETTINGS.items():
            self.config[key] = value
        self.init_model(model_name)