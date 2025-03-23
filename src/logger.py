import logging  # Importing logging module for logging messages
import os  # Importing os module for file and directory operations
from datetime import datetime  # Importing datetime module to generate timestamped log files

# Generating a log file name with the current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Defining the directory path to store log files
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)  # Creating the logs directory if it does not exist

# Defining the full path of the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configuring the logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Setting the log file path
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Defining log format
    level=logging.INFO,  # Setting logging level to INFO
)