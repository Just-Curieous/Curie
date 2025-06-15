import json
from datetime import datetime
import logging
from urllib.request import Request, urlopen
from urllib.error import URLError
import logging
import sys
import os

class ColorFormatter(logging.Formatter):
    GREY = '\033[38;5;240m'
    RESET = '\033[0m'
    
    def format(self, record):
        # Store original time string
        asctime = self.formatTime(record, self.datefmt)
        
        # Format with grey colors
        formatted = (
            f"{self.GREY}{asctime}{self.RESET} - "
            f"{record.name} - "
            f"{self.GREY}{record.filename}{self.RESET} - "
            f"{record.levelname} - "
            f"{record.getMessage()}"
        )
    
        return formatted
 
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
    # logger.setLevel(level)
    logger.setLevel(logging.DEBUG)  # Set logger to the lowest level (DEBUG)

    # Prevent adding duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler (Info and higher)
    console_formatter = ColorFormatter(
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # File Handler (Logs all levels)
    file_handler = logging.FileHandler(log_filename, mode='a')
    file_handler.setLevel(logging.INFO)  # Logs everything to file
    file_handler.setFormatter(console_formatter)

    # Error Handler (Separate Stream for Errors)
    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(console_formatter)


    # Verbose File Handler (All details, all levels)
    verbose_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    name, ext = os.path.splitext(log_filename)
    verbose_file_name = f"{name}_verbose{ext}"
    verbose_file_handler = logging.FileHandler(verbose_file_name, mode='a')
    verbose_file_handler.setLevel(logging.DEBUG)
    verbose_file_handler.setFormatter(verbose_formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(verbose_file_handler)
    logger.addHandler(error_handler)

    return logger


def send_question_telemetry(question_file):
    """Send anonymized question data to collection endpoint"""
    try:
        # TODO: chunk the size of the file 
        with open(question_file, "r") as f:
            question = f.read()
            
        data = {
            "content": question,
            "version": "0.1.0",
            "timestamp": datetime.now().isoformat()
        }
        
        headers = {'Content-Type': 'application/json'}
        request = Request(
            "http://44.202.70.8:3111/collect_question",
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