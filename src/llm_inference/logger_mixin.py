import logging
from functools import cached_property

class LoggingMixin:
    @cached_property
    def logger(self):
        return logging.getLogger(f'{self.__class__.__module__}.{self.__class__.__name__}')
