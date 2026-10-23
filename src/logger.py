import logging
import os
from src.config import LOGS_DIR

def get_logger(name: str) -> logging.Logger:
    """
    Creates and configures a dual file & console logger.
    
    Args:
        name: Subsystem or script name for logger tagging.
        
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        
        # File handler in logs directory
        log_file = os.path.join(LOGS_DIR, f'{name}.log')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Console stream handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger
