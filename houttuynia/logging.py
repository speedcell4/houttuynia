import sys
from pathlib import Path

from logbook import FileHandler, StreamHandler
from logbook import CRITICAL, DEBUG, ERROR, INFO, NOTICE
from logbook import critical, debug, error, info, notice

__all__ = [
    'debug', 'info', 'notice', 'error', 'critical',
    'DEBUG', 'INFO', 'NOTICE', 'ERROR', 'CRITICAL',
    'push_stream_handler',
    'push_file_handler',
]


def push_stream_handler(stream=sys.stdout, level: int = NOTICE, encoding: str = 'utf-8') -> StreamHandler:
    handler = StreamHandler(stream=stream, level=level, encoding=encoding)
    handler.push_application()
    return handler


def push_file_handler(file: Path, level: int = NOTICE, encoding: str = 'utf-8') -> FileHandler:
    handler = FileHandler(file.expanduser().absolute().__str__(), level=level, encoding=encoding)
    handler.push_application()
    return handler
