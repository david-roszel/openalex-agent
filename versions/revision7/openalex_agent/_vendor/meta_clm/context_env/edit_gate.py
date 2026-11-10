# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fit vs shrink: which rule is in force, and the frozen rejection wording.

The rejection note is prompt text shown to the model; its wording must not
change.

The ``CLM_EDIT_GATE`` env var (``fit`` or ``shrink``) selects the rule without
an agent kwarg. Precedence: explicit kwarg over env var over default (fit).
"""

from __future__ import annotations

import os
from typing import Any

SHRINK_RULE = "A compaction must SHRINK context"


_GATES = {"fit": True, "shrink": False, "1": True, "true": True, "yes": True, "on": True,
          "0": False, "false": False, "no": False, "off": False}


def _parse_gate(value: Any, source: str) -> bool:
    """``fit`` / ``shrink`` (or a boolean: true = fit, false = shrink) -> allow growth."""
    if isinstance(value, bool):
        return value
    key = str(value).strip().lower()
    if key not in _GATES:
        raise ValueError(f"{source} must be fit or shrink (or a boolean), got {value!r}")
    return _GATES[key]


def resolve_allow_edit_growth(passed: Any, env: str | None = None) -> bool:
    """``passed`` is the agent kwarg (None = unset). ``env`` defaults to CLM_EDIT_GATE.

    Both accept ``fit`` or ``shrink``; the kwarg also accepts a boolean
    (true = fit, false = shrink)."""
    if passed is not None:
        return _parse_gate(passed, "allow_edit_growth")
    raw = env if env is not None else os.environ.get("CLM_EDIT_GATE", "fit")
    return _parse_gate(raw or "fit", "CLM_EDIT_GATE")


def fit_rule(limit: int) -> str:
    return f"An edit must FIT the {limit}-token limit"


def may_grow(*, allow_growth: bool, limit: int, after: int) -> bool:
    """Growth is allowed under the fit gate when the result still fits.

    With no budget there is no 'fits' to test, so growth stays refused.
    """
    return bool(allow_growth and limit and after <= limit)


def reject_rule(*, allow_growth: bool, limit: int) -> str:
    if allow_growth and limit:
        return fit_rule(limit)
    return SHRINK_RULE


def rejected_note(*, baseline: int, after: int, before: int, rule: str) -> str:
    return (
        f"\n[LIVE_CTX_MAIN.txt: edit REJECTED — it GREW context "
        f"~{baseline}->{after} tokens, so it was NOT applied (still ~{before}). "
        f"{rule}: you likely duplicated/appended "
        f"content — replace stale text with a SHORTER summary instead.]"
    )
