import cProfile, pstats, io
from test_config import *

if LOG_PROFILE:
    import logging
    logger= logging.getLogger(__name__)





class Profiler():
    def __init__(self):
        self.pr = cProfile.Profile()

    def __enter__(self):
        self.pr.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(self.pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        profile = s.getvalue()
        if LOG_PROFILE:
            logger.info(profile)
        else:
            print(profile)
