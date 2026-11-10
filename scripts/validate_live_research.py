"""Copy selected V3 diagnostics here and run only offline tests.

Never executes a benchmark campaign or imports another checkout in place.
Frozen protocols, ledgers, approvals and running processes remain read-only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TESTS = ["test_multihorizon.py", "test_graph_visible_v2.py", "test_final_context_capacity.py", "test_terra_v2_accounting.py"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/ui-audit/v3-offline")
    parser.add_argument("--process-start-method", choices=("fork", "forkserver", "spawn"),
        help="Test harness only: some sandboxes cannot bind Python's forkserver socket")
    args = parser.parse_args()
    source, output = args.v3_root.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError("Choose a new output directory inside the isolated checkout")
    snapshot = output / "snapshot"
    snapshot.mkdir(parents=True)
    for folder in ("openalex_agent", "protocols", "data/v3-cases-draft-v2", "engineering/provider-docs"):
        shutil.copytree(source / folder, snapshot / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("engineering/pricing-verification.json", "requirements.lock.txt", "pyproject.toml"):
        (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, snapshot / name)
    (snapshot / "tests").mkdir()
    for name in TESTS:
        shutil.copyfile(source / "tests" / name, snapshot / "tests" / name)
    hashes = {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob("*") if p.is_file()}
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "read_only_source": str(source),
        "snapshot_sha256": hashes, "source_matches_at_copy": all(digest(source / n) == h for n, h in hashes.items()),
        "tests": TESTS, "network": "Denied by inherited Python audit hook", "paid_inference": False,
        "external_runner_executed": False, "external_files_written": False}
    (output / "snapshot-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(ROOT / "integration/live_offline_guard") + os.pathsep + str(snapshot))
    pytest_args = ["-q", "-p", "no:cacheprovider",
        "--basetemp", str(output / "test-tmp"), "--junitxml", str(output / "tests.xml"),
        *["tests/" + name for name in TESTS]]
    command = [sys.executable, "-m", "pytest", *pytest_args]
    if args.process_start_method:
        command = [sys.executable, "-c", "import multiprocessing,sys; multiprocessing.set_start_method(sys.argv[1]); "
            "import pytest; raise SystemExit(pytest.main(sys.argv[2:]))", args.process_start_method, *pytest_args]
    manifest["test_process_start_method"] = args.process_start_method or "Python default"
    with (output / "tests.log").open("w") as stream:
        result = subprocess.run(command, cwd=snapshot, env=env, stdout=stream, stderr=subprocess.STDOUT)
    manifest.update(exit_code=result.returncode,
        original_source_changes_during_tests=[n for n, h in hashes.items() if digest(source / n) != h])
    (output / "validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"exit_code": result.returncode, "output": str(output)}))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
