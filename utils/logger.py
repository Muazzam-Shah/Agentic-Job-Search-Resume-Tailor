"""
Simple logging utility for Job Hunter
"""

import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

def setup_logger(name: str = "JobHunter", level: str = None) -> logging.Logger:
    """
    Set up a logger with file and console handlers
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    # Get log level from environment or use provided level
    if level is None:
        level = os.getenv('LOG_LEVEL', 'INFO')
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler with UTF-8 encoding
    log_file = os.path.join(LOGS_DIR, f'job_hunter_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler with UTF-8 encoding and error handling
    import sys
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Set console encoding to UTF-8 if possible (Windows compatibility)
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        # Fallback: console will use 'replace' error handler automatically
        pass
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Create default logger
logger = setup_logger()
