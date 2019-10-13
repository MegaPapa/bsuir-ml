import time

from functools import wraps

from util.logger import LoggerBuilder

logger = LoggerBuilder().with_name("profiler").build()


def timed(func):
    """This decorator prints the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time_ns()
        result = func(*args, **kwargs)
        end = time.time_ns()
        logger.debug("Method '{}' ran in {} ms".format(func.__name__, round((end - start) / 1000000, 2)))
        result = func(*args, **kwargs)
        return result

    return wrapper
