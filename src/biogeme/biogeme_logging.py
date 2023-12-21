""" Interface for the use of the loggers in Biogeme

:author: Michel Bierlaire
:date: Thu Mar 23 17:53:22 2023

"""
import logging

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

loggers = (
    logging.getLogger('biogeme'),
    logging.getLogger('biogeme_optimization'),
)


def get_screen_logger(level: int = WARNING) -> logging.Logger:
    """Obtain a screen logger

    :param level: level of verbosity of the logger
    :type level: int
    """
    for logger in loggers:
        # This has to be set to the lower level: DEBUG so that it does not
        # supersed the handler
        logger.setLevel(DEBUG)
        formatter_debug = logging.Formatter(
            '[%(levelname)s] %(asctime)s %(message)s <%(filename)s:%(lineno)d>'
        )
        formatter_normal = logging.Formatter('%(message)s ')
        formatter = formatter_debug if level == DEBUG else formatter_normal
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return loggers[0]


def get_file_logger(filename: str, level: int = WARNING) -> logging.Logger:
    """Obtain a file logger

    :param filename: name of the file. Extension .log recommended.
    :type filename: str

    :param level: level of verbosity of the logger
    :type level: int
    """
    for logger in loggers:
        # This has to be set to the lower level: DEBUG so that it does not
        # supersed the handler
        logger.setLevel(DEBUG)
        formatter = logging.Formatter(
            '[%(levelname)s] %(asctime)s %(message)s <%(filename)s:%(lineno)d>'
        )
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return loggers[0]
