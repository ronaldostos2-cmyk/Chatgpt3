
import logging, sys, json
from pythonjsonlogger import jsonlogger

def setup_logging(level='INFO'):
    logger = logging.getLogger()
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
