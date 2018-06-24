from datetime import datetime
import subprocess
import hashlib
from pathlib import Path
from typing import Any
import json
import socket
import os
import pickle
import functools

from houttuynia import log_system as logging

__all__ = [
    'ensure_output_dir',
    'options_json', 'options_dump',

    'git_hash', 'datetime_hash', 'options_hash',
    'experiment_hash',

    'serialization',
]


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def git_hash(command: str = 'git rev-parse HEAD', shell: bool = True, encoding: str = 'utf-8') -> str:
    commit = subprocess.check_output(command, shell=shell)
    return str(commit, encoding=encoding).strip()


def datetime_hash(time_format: str = r'%y-%m%d-%H%M%S') -> str:
    return datetime.strftime(datetime.now(), time_format).strip()


def get_pid() -> int:
    return os.getpid()


def get_hostname() -> str:
    return socket.gethostname()


def options_json(**options: Any) -> str:
    return json.dumps({
        key: f'{options[key]}' for key in options.keys()
    }, indent=2, ensure_ascii=False, sort_keys=True)


def options_hash(hash_fn=hashlib.sha1, encoding: str = 'utf-8', **options: Any) -> str:
    bytes_repr = bytes(options_json(**options), encoding=encoding)
    return hash_fn(bytes_repr).hexdigest()


def options_dump(path: Path, __json_name: str = 'options.json', __encoding: str = 'utf-8', **options: Any) -> None:
    with (path / __json_name).open(mode='w', encoding=__encoding) as fp:
        return print(options_json(pid=get_pid(), hostname=get_hostname(), **options), file=fp)


def experiment_hash(hash_length: int = 8, **options: Any) -> str:
    return f'{datetime_hash()}-{git_hash()[:hash_length]}-{options_hash(**options)[:hash_length]}'


# TODO receive function to generate a Path
def serialization(path: Path):
    def fn_wrap(func):
        @functools.wraps(func)
        def arg_wrap(*args, **kwargs):
            try:
                logging.notice(f'loading data from {path}')
                with path.open(mode='rb') as fp:
                    ret = pickle.load(fp)
                return ret
            except (FileNotFoundError, AttributeError):
                ret = func(*args, **kwargs)
                logging.notice(f'dumping data to {path}')
                with path.open(mode='wb') as fp:
                    pickle.dump(ret, fp)
                return ret

        return arg_wrap

    return fn_wrap


if __name__ == '__main__':
    print(experiment_hash(b=2, a=2))
