"""
Logging utilities for EEG classification.
"""
import logging
import os
import sys
from typing import Optional

# Setup logging
def setup_logger(name: str = 'eeg_classification', 
                level: int = logging.INFO, 
                log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up and configure a logger.
    
    Args:
        name: Name of the logger
        level: Logging level (default: INFO)
        log_file: Path to log file (optional)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if a log file is specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create a default logger
logger = setup_logger() 