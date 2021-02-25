import logging
import os
from importlib import reload


def start_logging(log_file, logging_info):
    if os.path.exists(log_file):
        os.remove(log_file)    
    reload(logging)
    logger = logging.getLogger()
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    logging.info(f'{logging_info} STARTS.')

    
def end_logging(logging_info):
    logging.info(f'{logging_info} ENDS.')
    logging.shutdown()
    reload(logging)
    