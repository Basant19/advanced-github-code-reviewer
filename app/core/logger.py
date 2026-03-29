#E:\advanced-github-code-reviewer\app\core\logger.py
import logging
import os
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_PATH = os.path.join(LOGS_DIR, LOG_FILE)

logging.basicConfig(
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler() # This ensures you see it in the terminal
    ]
)

def get_logger(name: str):
    return logging.getLogger(name)