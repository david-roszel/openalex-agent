"""Inspect publication candidates or exact staged blobs without printing secrets.

This is a conservative local check, not a replacement for manual review or a
GitHub secret scanner. It never initializes Git, stages files or contacts a host.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


DIRECTORIES = ("openalex_agent", "tests", "benchmarks", "docs", "scripts", ".github")
ROOT_FILES = (".gitignore", ".env.example", "README.md", "pyproject.toml",
              "requirements.txt", "requirements.lock.txt", "requirements-ui.lock.txt", "requirements-gemini.txt",
              "main.py", "app.py", "launch-dashboard.sh", "PUBLISHING.md", "PUBLIC_MANIFEST.json", "LICENSE", "CITATION.cff",
              ".streamlit/config.toml")
LOCAL_DIRECTORIES = {"data", ".venv", "venv", ".git", ".aws", ".codex", ".agents",
                     ".vscode", ".idea", "__pycache__", ".pytest_cache", "build", "dist"}
PATTERNS = {
    "provider-key pattern": re.compile(rb"(?:sk-[A-Za-z0-9_-]{20,}|gsk_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{30,})"),
    "Git-host token pattern": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|glpat-[A-Za-z0-9_-]{20,})"),
    "AWS access-key pattern": re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "private-key header": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def local_only(name):
    path = Path(name)
    return (any(p in LOCAL_DIRECTORIES or p.endswith(".egg-info") for p in path.parts)
            or path.name != ".env.example" and (path.name == ".env" or path.name.startswith(".env."))
            or path.suffix in (".pem", ".key", ".p12", ".pfx", ".pyc", ".log", ".db")
            or ".sqlite" in path.name
            or path.name.startswith(("credentials", "service-account")) and path.suffix == ".json"
            or ".streamlit" in path.parts and path.name.startswith("secrets"))


def configured_secrets(root):
    """Values never enter a report, exception message or command argument."""
    values = {}
    for name, value in os.environ.items():
        if re.search(r"(?:API_KEY|TOKEN|SECRET|PASSWORD|EMAIL)$", name) and len(value) >= 8:
            values[name] = value.encode()
    for path in (root / ".env", root / ".streamlit/secrets.toml"):
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.lstrip().startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                value = value.strip().strip("\"'")
                if re.search(r"(?:KEY|TOKEN|SECRET|PASSWORD|EMAIL)$", name.strip(), re.I) and len(value) >= 8:
                    values[name.strip()] = value.encode()
    return values


def source_candidates(root):
    for name in ROOT_FILES:
        path = root / name
        if path.is_file() or path.is_symlink():
            yield name, path
    for directory in DIRECTORIES:
        for path in sorted((root / directory).rglob("*")):
            name = path.relative_to(root).as_posix()
            if (path.is_file() or path.is_symlink()) and not local_only(name):
                yield name, path


def git_command(root, *arguments):
    executable = shutil.which("git")
    if executable is None:
        raise RuntimeError("Git is not installed; exact staged/index checks are unavailable")
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    environment.update(GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    result = subprocess.run([executable, "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null",
                             "-C", str(root), *arguments], capture_output=True, timeout=30, env=environment)
    if result.returncode:
        raise RuntimeError("Git inspection failed; initialize/check the repository with your local Git installation")
    return result.stdout


def staged_candidates(root):
    records = git_command(root, "ls-files", "--stage", "-z")
    for record in records.split(b"\0"):
        if not record:
            continue
        header, name = record.split(b"\t", 1)
        mode, oid, stage = header.split()
        if stage != b"0":
            raise RuntimeError("Resolve index conflicts before publication")
        yield os.fsdecode(name), mode == b"120000", git_command(root, "cat-file", "blob", oid.decode("ascii"))


def audit(root, *, staged=False):
    root = Path(root).resolve()
    secrets = configured_secrets(root)
    inventory, concerns = [], []
    candidates = staged_candidates(root) if staged else (
        (name, path.is_symlink(), b"" if path.is_symlink() else path.read_bytes())
        for name, path in source_candidates(root))
    for name, symlink, data in candidates:
        inventory.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        if symlink:
            concerns.append({"path": name, "type": "symlink requires explicit review"})
        if local_only(name):
            concerns.append({"path": name, "type": "local-only artifact included"})
        if len(data) >= 10 * 1024 * 1024:
            concerns.append({"path": name, "type": "large publication file (10 MiB or more)"})
        for label, pattern in PATTERNS.items():
            if pattern.search(data):
                concerns.append({"path": name, "type": label})
        for label, secret in secrets.items():
            if secret in data:
                concerns.append({"path": name, "type": "configured sensitive value: " + label})
    required = [".gitignore", ".env.example", "README.md", "pyproject.toml",
                "tests/fixtures/openalex-list.synthetic.json",
                "tests/fixtures/synthetic-corpus/manifest.json",
                "tests/fixtures/synthetic-corpus/papers.jsonl",
                "openalex_agent/tokenizer_data/o200k_base.tiktoken",
                "openalex_agent/tokenizer_data/LICENSE"]
    present = {item["path"] for item in inventory}
    for name in required:
        if name not in present:
            concerns.append({"path": name, "type": "essential publication file missing"})
    return {"mode": "exact Git index blobs" if staged else "source candidate allowlist; not a Git index check",
            "git_installed": shutil.which("git") is not None,
            "sensitive_values_disclosed": False, "files": len(inventory),
            "total_bytes": sum(item["bytes"] for item in inventory),
            "concerns": concerns, "inventory": inventory,
            "scope": "Working tree/index only; old Git history, external backups and unknown secret formats require separate review"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Inspect exact index blobs; requires Git")
    parser.add_argument("--json", type=Path, help="Optional local JSON inventory/report destination")
    args = parser.parse_args()
    try:
        report = audit(Path(__file__).resolve().parents[1], staged=args.staged)
    except (RuntimeError, OSError, subprocess.TimeoutExpired):
        print("Publication audit unavailable: Git/index/files could not be safely inspected. No approval is implied.")
        raise SystemExit(2) from None
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "inventory"}, indent=2))
    if report["concerns"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
