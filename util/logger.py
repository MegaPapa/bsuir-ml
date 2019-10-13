import logging


class LoggerBuilder:
    def __init__(self):
        self.name = "application"
        self.level = "DEBUG"
        self.format = "%(asctime)s - %(name)-20s - [%(levelname)-5s] -- %(message)s"

    def with_name(self, logger_name):
        self.name = logger_name
        return self

    def with_format(self, logger_format):
        self.format = logger_format
        return self

    def with_level(self, logger_level):
        self.level = logger_level
        return self

    def build(self):
        logger = logging.getLogger(self.name)

        # Misc logger setup so a debug log statement gets printed on stdout.
        logger.setLevel(self.level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(self.format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger