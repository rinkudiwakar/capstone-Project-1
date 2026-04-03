import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime


# constant for log configurations
LOG_DIR = 'logs'
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE= 5*1024*1024
BACKUP_COUNT = 3 


# Construct log file path
root_dir= os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
log_dir_path= os.path.join(root_dir,LOG_DIR)
os.makedirs(log_dir_path,exist_ok=True)
log_file_path= os.path.join(log_dir_path,LOG_FILE)

#configure logger function
def configure_logger():
    """
    Configures logging with a rotating file handler and a console handler.
    """
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Define formatter
    formatter=logging.Formatter("[%(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    #File Handler with Rotation
    file_handler= RotatingFileHandler(log_file_path,maxBytes=MAX_LOG_SIZE,backupCount=BACKUP_COUNT)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    #console handler
    console_handler= logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    #add handler to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
 
# Configure the logger
logger=configure_logger()


    
    
    


    
    
    
    
    


