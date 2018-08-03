from importlib import reload
import src.trainer
reload(src.trainer)
from src.trainer import Trainer
coach = Trainer('uttt')
coach.learn()

import src.games.uttt.game as game
from importlib import reload
reload(game)
import numpy as np
mb = game.UTTTState()
mb.to_input()