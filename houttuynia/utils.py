from pathlib import Path
from typing import Union


def ensure_path(path: Union[str, Path]) -> Path:
    if not isinstance(path, Path):
        path = Path(path)
    return path


a = Path('.')
print(Path(a).__class__)
