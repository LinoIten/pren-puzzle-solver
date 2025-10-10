"""
Einfaches Logging für PREN Puzzle Solver
"""

import logging
import sys

def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Erstellt einen einfachen Logger
    
    Args:
        name: Name des Loggers
        level: Logging-Level (default: INFO)
        
    Returns:
        Logger
    """
    logger = logging.getLogger(name)
    
    # Verhindere doppelte Handler
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Console Handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Einfaches Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger
