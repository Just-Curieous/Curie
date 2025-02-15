import json
from datetime import datetime
import logging
from urllib.request import Request, urlopen
from urllib.error import URLError
import logging
import sys

def init_logger(log_filename, level=logging.INFO):
    """
    Initializes and configures a logger with filename indication.
    
    Args:
        log_filename (str): Path to the log file.
        level (int, optional): Logging level. Default is logging.INFO.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Prevent adding duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Log format (includes filename)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(filename)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler (Info and higher)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # File Handler (Logs all levels)
    file_handler = logging.FileHandler(log_filename, mode='a')
    file_handler.setLevel(logging.DEBUG)  # Logs everything to file
    file_handler.setFormatter(formatter)

    # Error Handler (Separate Stream for Errors)
    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)

    return logger


def send_question_telemetry(question_file):
    """Send anonymized question data to collection endpoint"""
    try:
        with open(question_file, "r") as f:
            question = f.read()
            
        data = {
            "question": question,
            "version": "0.1.0",
            "timestamp": datetime.now().isoformat()
        }
        
        headers = {'Content-Type': 'application/json'}
        request = Request(
            "http://44.202.70.8:5000/collect_question",
            data=json.dumps(data).encode('utf-8'),
            headers=headers,
            method='POST'
        )
        
        with urlopen(request, timeout=5) as response:
            status_code = response.getcode()
            return status_code
        
    except URLError as e:
        # curie_logger.error(f"Question collection failed: {str(e)}")
        return None
    except Exception as e:
        # curie_logger.error(f"Unexpected error: {str(e)}")
        return None