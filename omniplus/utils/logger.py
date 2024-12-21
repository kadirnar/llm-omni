import logging
from typing import Optional

from termcolor import colored

LOGGER_COLORS = {"PaliGemma": "blue", "TestScript": "green", "default": "white"}


class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored logging output."""

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        # Get color based on logger name
        color = LOGGER_COLORS.get(record.name, LOGGER_COLORS["default"])
        record.name = colored(record.name, color)
        record.msg = colored(record.msg, color)
        return super().format(record)


def setup_logger(name: str, level: Optional[int] = logging.INFO) -> logging.Logger:
    """
    Set up a colored logger instance.

    Args:
        name (str): Name of the logger
        level (int, optional): Logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)

        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
