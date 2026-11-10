"""Build an allowlisted, deterministic standalone submission without local secrets."""
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build(root=ROOT, output=None):
    root = Path(root).resolve()
    output = Path(output) if output else root / "dist"
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for name in ("README.md", ".env.example", "pyproject.toml", "requirements.lock.txt", "requirements-ui.lock.txt",
                 "requirements.txt", "live_app.py", "app.py", "legacy_app.py", "main.py", "launch-live-research.sh", "launch-dashboard.sh", ".gitignore", ".streamlit/config.toml"):
        if (root / name).is_file():
            files.append(root / name)
    for name in ("openalex_agent", "tests", "benchmarks"):
        files += [p for p in (root / name).rglob("*") if p.is_file() and not p.is_symlink() and
            "__pycache__" not in p.parts and p.suffix not in (".pyc", ".sqlite", ".log")]
    for name in ("docs/submission", "integration/live_offline_guard"):
        files += [p for p in (root / name).rglob("*") if p.is_file() and not p.is_symlink() and
            p.suffix in (".md", ".json", ".tex", ".csv", ".py", ".xml", ".log") and
            p.name not in ("README-before-submission.md", "package-build-receipt.json")]
    # Keep utilities imported by regression tests; none are executed by packaging.
    files += [p for p in (root / "scripts").glob("*.py") if p.is_file() and not p.is_symlink()]
    secrets = []
    secret_names = ("OPENALEX_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
    if (root / ".env").is_file():
        from dotenv import dotenv_values
        secrets += [v for k, v in dotenv_values(root / ".env").items() if k in secret_names and v and len(v) >= 8]
    secrets += [os.environ[k] for k in secret_names if len(os.environ.get(k, "")) >= 8]
    contents = {}
    for path in sorted(set(files)):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Submission cannot follow links outside the isolated source tree")
        raw = path.read_bytes()
        if any(secret.encode() in raw for secret in secrets):
            raise ValueError("Configured credential detected; refusing to package file: " + relative)
        contents[relative] = raw
    manifest = {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)} for name, raw in contents.items()}
    receipt = {"configured_secret_scan": "PASS", "files": len(contents), "total_uncompressed_bytes": sum(len(v) for v in contents.values()),
        "excluded": [".env", "virtual environments", "databases", "data/", "private workspace metadata", "V3 project and experiment artifacts"],
        "release": "OpenAlex Live Research standalone submission"}
    archive = output / "openalex-live-research-submission.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as package:
        for name, raw in {**contents, "MANIFEST.json": (json.dumps(manifest, indent=2) + "\n").encode(),
                          "SECRET_SCAN.json": (json.dumps(receipt, indent=2) + "\n").encode()}.items():
            info = zipfile.ZipInfo("OpenAlex-LiveResearch/" + name, date_time=(2026, 11, 9, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if name.endswith(".sh") else 0o644) << 16
            package.writestr(info, raw)
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    receipt["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output / "SECRET_SCAN.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return archive, receipt


if __name__ == "__main__":
    path, receipt = build()
    print(json.dumps({"archive": str(path), **receipt}, indent=2))
