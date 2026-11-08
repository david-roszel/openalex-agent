# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Token counting, per-turn snapshots, and cache-friendly truncation.

All harness code that needs to know "how big is my context right now?" goes
through :func:`count_tokens` (exact tiktoken, falling back to a chars/4 estimate
when tiktoken is unavailable offline). :func:`record` persists the full retained
context each turn so a run can be replayed / graded turn-by-turn.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SNAPSHOT_DIRNAME = "context_snapshots"

_ENCODER: Any = None
_ENCODER_NAME: str | None = None


def get_encoder():
    """Return a cached tiktoken encoder, or ``None`` if tiktoken is unavailable.

    Defaults to OpenAI's ``o200k_base`` (GPT-4o/4.1 family); override with the
    ``TIKTOKEN_ENCODING`` env var, and fall back to ``cl100k_base``.
    """
    global _ENCODER, _ENCODER_NAME
    if _ENCODER is not None or _ENCODER_NAME == "_failed":
        return _ENCODER
    try:
        import tiktoken

        for name in (os.environ.get("TIKTOKEN_ENCODING") or "o200k_base", "cl100k_base"):
            try:
                _ENCODER = tiktoken.get_encoding(name)
                _ENCODER_NAME = name
                return _ENCODER
            except Exception:  # e.g. vocab download blocked offline
                continue
        _ENCODER_NAME = "_failed"
    except Exception:  # tiktoken not installed
        _ENCODER_NAME = "_failed"
    return _ENCODER


def _iter_text(messages: list[Any] | None):
    """Yield every text piece that contributes to the prompt sent to the model.

    Crucially this includes ``tool_calls[*].function.arguments`` (where a bash
    command lives): a bash-heavy agent stores most of its bytes there, so
    counting only ``content`` would undercount the real prompt by ~2x.
    """
    for msg in messages or []:
        if isinstance(msg, dict):
            reasoning = msg.get("reasoning_content")
            if isinstance(reasoning, str):
                yield reasoning
            content = msg.get("content")
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") if isinstance(tc, dict) else None
                if isinstance(fn, dict):
                    if fn.get("name"):
                        yield str(fn["name"])
                    if fn.get("arguments"):
                        yield str(fn["arguments"])
        else:
            content = msg
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    yield str(item.get("text") or item.get("content") or "")
                else:
                    yield str(item)


def approx_tokens(messages: list[Any] | None) -> int:
    """~4 chars/token estimate (fallback only, when tiktoken is unavailable)."""
    return (sum(len(t) for t in _iter_text(messages)) + 1) // 4


def count_tokens(messages: list[Any] | None) -> tuple[int, str]:
    """Exact tiktoken token count over all message text. Returns ``(tokens, method)``."""
    enc = get_encoder()
    if enc is None:
        return approx_tokens(messages), "approx_chars4"
    total = 0
    for text in _iter_text(messages):
        if text:
            total += len(enc.encode(text, disallowed_special=()))
    return total, f"tiktoken:{_ENCODER_NAME}"


def head_tail_truncate(text: str, keep_tokens: int, budget: int) -> str:
    """Keep the head and tail of ``text`` totalling ~``keep_tokens``, eliding the
    middle with a marker. Cache-friendly: only ever applied to the NEWEST message
    so earlier (cached) messages are never disturbed."""
    enc = get_encoder()
    if enc is None:
        approx_chars = max(keep_tokens * 4, 256)
        if len(text) <= approx_chars:
            return text
        half = approx_chars // 2
        elided = len(text) - 2 * half
        return (
            f"{text[:half]}\n\n[... ~{elided} chars elided to respect the "
            f"{budget}-token context cap; narrow the command, or compact stale "
            f"context with your context-editing tools ...]\n\n{text[-half:]}"
        )
    toks = enc.encode(text, disallowed_special=())
    if len(toks) <= keep_tokens:
        return text
    keep = max(keep_tokens - 48, 128)  # ~48 tokens for the marker; floor 128
    head_n = keep // 2
    tail_n = keep - head_n
    elided = len(toks) - head_n - tail_n
    head = enc.decode(toks[:head_n])
    tail = enc.decode(toks[-tail_n:])
    return (
        f"{head}\n\n[... {elided} tokens elided to respect the {budget}-token "
        f"context cap; narrow the command, or compact stale context with your "
        f"context-editing tools ...]\n\n{tail}"
    )


def annotate(logs_dir: Path | str, step: int, extra: dict[str, Any]) -> None:
    """Merge extra fields into an EXISTING per-turn snapshot without rewriting its
    ``messages`` (best-effort; never raises). Used to add e.g. ``sent_tokens`` (the
    post-compaction context actually sent to the model) after ``record`` captured the
    full pre-compaction context for the transcript."""
    try:
        f = Path(logs_dir) / _SNAPSHOT_DIRNAME / f"turn-{step:04d}.json"
        if not f.exists():
            return
        d = json.loads(f.read_text(encoding="utf-8"))
        d.update(extra)
        f.write_text(json.dumps(d, indent=2, default=str) + "\n", encoding="utf-8")
    except Exception as exc:  # logging side-effect must not crash a run
        logger.debug("annotate snapshot failed at step %s: %s", step, exc)


def record(
    logs_dir: Path | str,
    step: int,
    messages: list[Any],
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist one per-turn snapshot of the FULL retained context (best-effort;
    never raises so logging can't crash a run)."""
    try:
        d = Path(logs_dir) / _SNAPSHOT_DIRNAME
        d.mkdir(parents=True, exist_ok=True)
        tokens, method = count_tokens(messages)
        payload: dict[str, Any] = {
            "step": step,
            "tokens": tokens,
            "token_method": method,
            "n_messages": len(messages or []),
        }
        if extra:
            payload.update(extra)
        payload["messages"] = messages
        (d / f"turn-{step:04d}.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
        )
    except Exception as exc:  # logging side-effect must not crash a run
        logger.debug("context snapshot failed at step %s: %s", step, exc)
