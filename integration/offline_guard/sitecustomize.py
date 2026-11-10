"""Offline validation: deny real IP connections and writes outside this checkout.

Inherited by subprocesses. PYTHONDONTWRITEBYTECODE avoids imported-source writes.
No credentials or live-accounting initialization is needed for these tests.
"""
import os
from pathlib import Path
import socket
import sys

WORKSPACE = Path(__file__).resolve().parents[2]


def audit(event, args):
    if event == "socket.connect" and args[0].family in (socket.AF_INET, socket.AF_INET6):
        raise RuntimeError("OFFLINE VALIDATION: real IP networking prohibited")
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        if (args[2] or 0) & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            path = Path(os.fsdecode(args[0])).resolve()
            if not path.is_relative_to(WORKSPACE) and not path.is_relative_to("/tmp") and str(path) != "/dev/null":
                raise RuntimeError("OFFLINE VALIDATION: write outside isolated checkout prohibited")


sys.addaudithook(audit)
