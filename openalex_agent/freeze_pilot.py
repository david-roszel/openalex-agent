"""Byte-preserving, offline snapshots of explicitly selected user-owned files."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .models import utcnow


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(sources: dict[str, Path], output: Path, *, forbidden: tuple[str, ...] = ()) -> dict:
    """Create once, stage atomically, and checksum every byte. Never overwrite."""
    if output.exists():
        raise ValueError("Frozen snapshot already exists; use a new version")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pilot-freeze-", dir=output.parent) as temporary:
        stage = Path(temporary) / "snapshot"
        stage.mkdir()
        files = {}
        for name, source in sorted(sources.items()):
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or name == "manifest.json":
                raise ValueError("Unsafe snapshot path")
            if source.is_symlink() or not source.is_file():
                raise ValueError("Snapshot sources must be regular files")
            data = source.read_bytes()
            if any(secret.encode() in data for secret in forbidden if secret):
                raise ValueError("A source contains a configured credential; snapshot cancelled")
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            files[relative.as_posix()] = {"sha256": digest(target), "bytes": len(data)}
        manifest = {"schema": 1, "created_at": utcnow(), "files": files,
                    "policy": "Preserved bytes; no prompt, setting, corpus or output is rewritten."}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # rename refuses replacement of an existing nonempty snapshot directory.
        if output.exists():
            raise ValueError("Frozen snapshot already exists; use a new version")
        stage.rename(output)
    return {"schema": 1, "snapshot": str(output), "manifest_sha256": digest(output / "manifest.json"),
            "files": len(files), "bytes": sum(f["bytes"] for f in files.values())}


def verify(receipt: dict) -> dict:
    root = Path(receipt["snapshot"])
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or digest(manifest_path) != receipt["manifest_sha256"]:
        raise ValueError("Snapshot manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["files"]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if any(p.is_symlink() for p in root.rglob("*")) or actual != set(expected) | {"manifest.json"}:
        raise ValueError("Snapshot file inventory mismatch")
    for name, record in expected.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe snapshot path")
        target = root / path
        if target.stat().st_size != record["bytes"] or digest(target) != record["sha256"]:
            raise ValueError(f"Snapshot file checksum mismatch: {name}")
    return {"verified": True, "files": len(expected), "manifest_sha256": receipt["manifest_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify"))
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--output", default="data/frozen/snapshot")
    parser.add_argument("--sources", help="JSON mapping of snapshot-relative names to explicitly selected source files")
    args = parser.parse_args()
    receipt_path = Path(args.receipt)
    if args.command == "verify":
        print(json.dumps(verify(json.loads(receipt_path.read_text(encoding="utf-8")))))
        return
    if receipt_path.exists():
        raise ValueError("Receipt already exists; use a new version")
    if not args.sources:
        parser.error("freeze requires an explicit --sources mapping; no project directories are copied implicitly")
    mapping = json.loads(Path(args.sources).read_text())
    if not isinstance(mapping, dict) or not mapping or any(not isinstance(k, str) or not isinstance(v, str) for k,v in mapping.items()):
        raise ValueError("Sources must map snapshot-relative names to file paths")
    sources = {name: Path(path) for name, path in mapping.items()}
    # Load only for an exclusion check; the actual .env and keys are never copied.
    from dotenv import dotenv_values
    import os
    configured = dotenv_values(".env")
    secrets = tuple(os.getenv(k) or configured.get(k) or "" for k in ("LLM_API_KEY", "OPENALEX_API_KEY"))
    receipt = snapshot(sources, Path(args.output), forbidden=secrets)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verify(receipt)))


if __name__ == "__main__":
    main()
