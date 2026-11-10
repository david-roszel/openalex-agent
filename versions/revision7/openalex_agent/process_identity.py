"""Linux process identities: boot ID + PID + start tick, never PID alone."""
import os
from pathlib import Path


def identity(pid=None):
    pid = os.getpid() if pid is None else int(pid)
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return None
        return {'pid': pid, 'start': fields[19],
                'boot': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except (OSError, ValueError, IndexError):
        return None


def alive(record):
    return isinstance(record, dict) and identity(record.get('pid', -1)) == record
