"""Per-run editable inference context; never an authoritative metadata store."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

from .corpus import atomic_json


def serialized(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class ContextStore:
    """A single writable text file, with immutable observation and edit records.

    Model requests are data, never shell commands or filesystem paths. Even a
    whole-file replacement changes only active_context.txt in this workspace.
    """

    def __init__(self, directory: Path, max_chars: int, sink=None):
        self.directory = Path(directory).absolute()
        if self.directory.resolve() != self.directory or self.directory.exists():
            raise ValueError("Context workspace must be a new, non-redirected directory")
        self.directory.mkdir(parents=True)
        self.max_chars = max_chars
        self.sink = sink or (lambda kind, value: None)
        self.path = self.directory / "active_context.txt"
        self.path.write_text("", encoding="utf-8")
        self.version = 0
        self.observations = []
        self.edits = []
        self._committed_hash = hashlib.sha256(b"").hexdigest()
        self._uncertain = False

    def _guard(self):
        if self.directory.resolve() != self.directory or self.path.is_symlink():
            raise ValueError("Context workspace redirection rejected")
        if any(p.is_symlink() for p in self.directory.iterdir()):
            raise ValueError("Context workspace contains a redirected file")

    def read(self) -> str:
        self._guard()
        return self.path.read_text(encoding="utf-8")

    def _write(self, text: str):
        self._guard()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def verify_committed(self):
        if self._uncertain or hashlib.sha256(self.read().encode()).hexdigest() != self._committed_hash:
            raise RuntimeError("Context commit is uncertain; stop this run and use a fresh replay")

    def archive(self, observation: dict) -> int:
        self._guard()
        index = len(self.observations)
        atomic_json(self.directory / f"observation-{index:05d}.json", observation)
        self.observations.append(observation)
        return index

    def recover(self, index: int) -> dict:
        if type(index) is not int or not 0 <= index < len(self.observations):
            raise ValueError("Unknown observation index")
        self._guard()
        return json.loads((self.directory / f"observation-{index:05d}.json").read_text(encoding="utf-8"))

    def transition(self, after: str, *, source: str, operation: dict, usage: dict | None = None,
                   error: str | None = None) -> dict:
        before = self.read()
        self.verify_committed()
        if error is None and len(after) > self.max_chars:
            error = "context_capacity_exceeded"
        number = self.version
        self.version += 1
        record = {"index": number, "source": source, "operation": operation, "requested": True,
                  "validated": error is None, "committed": False, "accepted": False,
                  "status": "requested", "error": error, "before": before, "after": before,
                  "requested_after": after, "usage": usage or {}}
        # Durable intent precedes any mutation. A missing terminal receipt means
        # interrupted/unknown, never success. These records are never rewritten.
        atomic_json(self.directory / f"operation-{number:05d}-requested.json", record)
        if error is None:
            atomic_json(self.directory / f"operation-{number:05d}-validated.json", {**record, "status": "validated"})
        interrupted = None
        if error is not None:
            record["status"] = "rejected"
        else:
            try:
                self._write(after)
                if self.read() != after:
                    raise OSError("Context readback mismatch")
                record.update(status="committed", committed=True, accepted=True)
            except (OSError, KeyboardInterrupt, SystemExit) as exc:
                interrupted = exc if isinstance(exc, (KeyboardInterrupt, SystemExit)) else None
                record.update(status="interrupted" if interrupted else "failed",
                              error="context_update_interrupted" if interrupted else "context_write_failed")
        actual = self.read()
        record.update(after=actual, applied=actual != before, before_chars=len(before), after_chars=len(actual),
                      before_sha256=hashlib.sha256(before.encode()).hexdigest(),
                      after_sha256=hashlib.sha256(actual.encode()).hexdigest())
        try:
            atomic_json(self.directory / f"transition-{number:05d}.json", record)
        except BaseException:
            self._uncertain = True
            raise
        if record["committed"]:
            self._committed_hash = record["after_sha256"]
        elif actual != before:
            self._uncertain = True
        self.edits.append(record)
        self.sink("context_edit" if source == "model" else "context_observation", record)
        if interrupted is not None:
            raise interrupted
        return record

    def edit(self, request: dict, *, usage: dict | None = None, fits=None) -> dict:
        before = self.read()
        try:
            if not isinstance(request, dict) or request.get("path") != "active_context.txt":
                raise ValueError("invalid_context_path")
            op = request.get("op")
            required = {"path", "op", "old", "new"} if op == "splice" else {"path", "op", "text"}
            if set(request) != required:
                raise ValueError("invalid_edit_fields")
            if op in ("replace", "append") and isinstance(request["text"], str):
                after = request["text"] if op == "replace" else before + request["text"]
            elif op == "splice" and all(isinstance(request[k], str) for k in ("old", "new")):
                if not request["old"] or before.count(request["old"]) != 1:
                    raise ValueError("edit_match_must_be_unique")
                after = before.replace(request["old"], request["new"], 1)
            else:
                raise ValueError("invalid_edit_operation")
        except ValueError as exc:
            return self.transition(before, source="model", operation=request, usage=usage, error=str(exc))
        if fits is not None and not fits(after):
            return self.transition(before, source="model", operation=request, usage=usage, error="context_request_capacity_exceeded")
        return self.transition(after, source="model", operation=request, usage=usage)
