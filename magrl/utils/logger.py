import logging
from logging.config import dictConfig

_LIBRARY_NAME = 'moaa'
_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
_DATE_FORMAT = "%m/%d/%Y %H:%M:%S"
_STREAM_HANDLER_LEVEL = "INFO"

DEFAULT_LOGGING_CONFIG = {
    "formatters": {
        _LIBRARY_NAME: {
            "class": "logging.Formatter",
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
        },
    },
    "handlers": {
        _LIBRARY_NAME: {
            "class": "logging.StreamHandler",
            "formatter": _LIBRARY_NAME,
            "level": _STREAM_HANDLER_LEVEL,
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        _LIBRARY_NAME: {
            "handlers": [_LIBRARY_NAME],
            "level": "DEBUG",
            "propagate": False,
        },
    },
    "version": 1,
    "disable_existing_loggers": False
}

dictConfig(DEFAULT_LOGGING_CONFIG)


def get_logger(name: str) -> logging.Logger:
    """
    在每个脚本里调用这个函数拿到logger
    """

    return logging.getLogger(name)


logger = get_logger(__name__)
