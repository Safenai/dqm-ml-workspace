"""CLI utilities for DQM-ML v2.

This module provides utilities for colored console output and custom
logging formatters.
"""

import logging

from typing_extensions import override


class Bcolors:
    """ANSI color codes for terminal output formatting."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


# TODO move logging in a dedicated module
class CustomFormatter(logging.Formatter):
    MSG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    LVL_COLOR = {
        logging.DEBUG: Bcolors.OKBLUE,
        logging.INFO: "",
        logging.WARNING: Bcolors.WARNING,
        logging.ERROR: Bcolors.FAIL,
        logging.CRITICAL: Bcolors.FAIL + Bcolors.BOLD,
    }

    @override
    def format(self, record: logging.LogRecord) -> str:
        color = self.LVL_COLOR.get(record.levelno, "")
        log_fmt = color + self.MSG_FMT + Bcolors.ENDC
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

    @classmethod
    def init_log(cls, level: int | str, format: str = MSG_FMT) -> None:
        # TODO : forward generic parameters of init_log to basicConfig
        logging.basicConfig(format=format, level=level)

        # We upgrade format for our color console formater
        cls.MSG_FMT = format

        # Adding color format to stream handler
        logger = logging.getLogger()
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(CustomFormatter())
