import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
import sys #Importing sys for the stream handler

def setup_logger():
    # Create logger
    logger = logging.getLogger('prem_elo_tracker')
    logger.setLevel(logging.DEBUG)

    # Create logs directory if it doesn't exist
    log_path = os.path.join("C:", "Users", "3fold", "Documents", "Prem ELO", "logs")
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # Create rotating file handler
    log_file = os.path.join(log_path, f'prem_elo_{datetime.now().strftime("%Y%m%d")}.log')
    handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )

    #Create Stream handler
    stream_handler = logging.StreamHandler(sys.stdout)

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter) #setting the formatter for the stream handler.

    logger.addHandler(handler)
    logger.addHandler(stream_handler) #Adding the stream handler to the logger.

    return logger

# Initialize the logger
logger = setup_logger()