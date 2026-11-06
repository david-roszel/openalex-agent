"""Read-only source revision metadata; never record remotes or Git configuration."""

import os
from pathlib import Path
import re
import shutil
import subprocess


def git_provenance(root=None):
    """Return an explicit unknown when Git/HEAD/status cannot be inspected.

    A commit identifies committed code only. Dirty working trees still require
    the existing source snapshots/checksums to identify the executed code.
    """
    record = {"commit": None, "dirty": None, "status": "git_unavailable"}
    executable = shutil.which("git")
    if executable is None:
        return record
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    # Inherited repository selectors must not point provenance at another repo.
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    environment.update(GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")

    def run(*arguments):
        return subprocess.run(
            [executable, "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null",
             "-C", str(root), *arguments],
            capture_output=True, text=True, timeout=5, check=False, env=environment,
        )

    try:
        repository = run("rev-parse", "--is-inside-work-tree")
        if repository.returncode != 0 or repository.stdout.strip() != "true":
            return {**record, "status": "not_a_repository"}
        revision = run("rev-parse", "--verify", "HEAD")
        commit = revision.stdout.strip()
        if revision.returncode != 0:
            return {**record, "status": "no_commits"}
        if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None:
            return {**record, "status": "git_error"}
        status = run("status", "--porcelain=v1", "--untracked-files=normal")
        return {"commit": commit, "dirty": bool(status.stdout) if status.returncode == 0 else None,
                "status": "recorded" if status.returncode == 0 else "status_unavailable"}
    except (OSError, subprocess.TimeoutExpired, UnicodeError):
        return {**record, "status": "git_error"}
