import sys
from pathlib import Path

import socket
import platform

REQUIRED_PYTHON_VERSION = (3, 6)

if sys.version_info < REQUIRED_PYTHON_VERSION:
    exit(f'currently, we only support Python {".".join(map(str, REQUIRED_PYTHON_VERSION))}')

host_name: str = socket.gethostname()
system_name: str = platform.system().lower()

app_dir = Path(__file__).expanduser().absolute().parent
project_dir = app_dir.parent
app_name = app_dir.name

out_dir = project_dir / 'out'

if not out_dir.exists():
    out_dir.mkdir(parents=True, exist_ok=True)
