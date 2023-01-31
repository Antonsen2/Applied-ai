import os
import asyncio
import logging
from enum import Enum
from networking import run_server

class LoggingLevel(Enum):
    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    @classmethod
    def get_level(cls, level: str):
        return cls.__members__.get(level, cls.NOTSET).value


LOGGER_NAME = "main"
FORMAT = "| %(asctime)s | %(levelname)s | %(name)s | %(message)s |"
logging.basicConfig(level=logging.INFO, format=FORMAT)
LOGGER = logging.getLogger(LOGGER_NAME)


def setup_logging():
    level = os.getenv('LOG_LEVEL', None)
    if level:
        logging_level = LoggingLevel.get_level(level.upper())
        logging.root.setLevel(logging_level)


def app():
    LOGGER.info("Starting app")
    loop = asyncio.new_event_loop()
    loop.run_until_complete(run_server())


if __name__ == '__main__':
    setup_logging()
    app()
