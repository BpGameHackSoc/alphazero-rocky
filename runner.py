from importlib import reload
import src.trainer
reload(src.trainer)
from src.trainer import Trainer
coach = Trainer('uttt')
coach.learn()
