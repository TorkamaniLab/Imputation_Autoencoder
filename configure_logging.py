import logging
import time
from multiprocessing_logging import install_mp_handler


# A new filter for measuring elapsed time
class ElapsedTimeFiler(logging.Filter):

    def __init__(self):
        super().__init__()
        self.start_time = time.time()

    def filter(self, record):
        elapsed_seconds = record.created - self.start_time
        hours = elapsed_seconds // 3600
        minutes = (elapsed_seconds % 3600) // 60
        seconds = elapsed_seconds % 60
        record.elapsed = "{}:{}:{:.4f}".format(int(hours), int(minutes), seconds)
        return True


elapsed_filter = ElapsedTimeFiler()


def configure_logging(logger, filename):
    # Prevent initialize handlers multiple times from inference wrapper script
    if logger.hasHandlers():
        for handler in logger.handlers.copy():
            logger.removeHandler(handler)
    # Adding elapsed time filter object
    logger.addFilter(elapsed_filter)
    logger.setLevel(logging.INFO)
    # Create file handler
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # Create formatters
    long_formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d - %(elapsed)s - %(process)d - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d_%H:%M:%S')
    short_formatter = logging.Formatter(
        fmt='%(elapsed)s - %(process)d - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d_%H:%M:%S')
    # Add formatter to the handlers
    ch.setFormatter(short_formatter)
    fh.setFormatter(long_formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    install_mp_handler(logger=logger)


def set_logger_to_debug(logger_obj):
    level = logging.DEBUG
    logger_obj.setLevel(level)
    for handler in logger_obj.handlers:
        handler.setLevel(level)
