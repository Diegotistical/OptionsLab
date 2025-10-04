# src/common/logging_config.py

import logging


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


""" Example usage:

from src.common.logging_config import setup_logging

setup_logging()  # call once on app start
logger = logging.getLogger(__name__)

"""
