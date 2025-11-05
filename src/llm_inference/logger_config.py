import logging
import sys

from llm_inference.settings import LOGGING_LEVEL

def configure_logging():
    # Configure the root logger for console output
    logger = logging.getLogger()
    logger.setLevel(LOGGING_LEVEL)

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOGGING_LEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
