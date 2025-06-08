"""Logging utilities for SoftAdaptX.

This module provides a centralized logging configuration for the SoftAdaptX package.
"""

import logging
import sys

# Create a logger
logger = logging.getLogger("softadaptx")

# Set default level to INFO
logger.setLevel(logging.INFO)

# Create console handler and set level to INFO
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Add formatter to console handler
console_handler.setFormatter(formatter)

# Add console handler to logger
logger.addHandler(console_handler)

# Prevent logging messages from being duplicated in the root logger
logger.propagate = False


def get_logger() -> logging.Logger:
    """Get the SoftAdaptX logger.

    Returns:
        logging.Logger: The SoftAdaptX logger.
    """
    return logger
