# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Whether the agent is allowed to end its own run.

Two modes, selected per trial by the ``finish_policy`` agent kwarg:

``submit`` (default)
    The system prompt tells the agent how to finish and the harness ends the
    run when the standalone marker command succeeds.

``open_ended``
    There is no finish command. The finish paragraph disappears from the system
    prompt, and if the agent runs the marker command anyway the harness treats
    it as an ordinary (no-op) shell command and keeps going. EdgeBench is scored
    on the BEST judged submission over a fixed horizon, so an agent that stops
    early only loses score -- the benchmark has no "done" state to reach.
"""

from __future__ import annotations

import re

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

#: The finish paragraph; fills the ``{{finish_instructions}}`` placeholder in
#: prompts.yaml in ``submit`` mode.
FINISH_INSTRUCTIONS = (
    "When you are finished, submit by running the standalone command\n"
    f"`echo {SUBMIT_MARKER}` (with nothing else in that command).\n"
    "After it runs you cannot continue."
)

MODES = ("submit", "open_ended")


class FinishPolicy:
    """Single switch for agent-issued termination."""

    __slots__ = ("mode",)

    def __init__(self, mode: str = "submit") -> None:
        m = (str(mode or "submit")).strip().lower()
        if m in ("", "1", "true", "submit"):
            m = "submit"
        elif m in ("open_ended", "open-ended", "openended"):
            m = "open_ended"
        else:
            raise ValueError(f"finish_policy must be one of {MODES}, got {mode!r}")
        self.mode = m

    @property
    def open_ended(self) -> bool:
        return self.mode == "open_ended"

    def is_finish_command(self, cmd: str) -> bool:
        """True only when the agent is allowed to end the run this way."""
        if self.open_ended:
            return False
        return is_submit_command(cmd)

    def prompt_fragment(self) -> str:
        """The system-prompt paragraph telling the agent how to finish."""
        if self.open_ended:
            return ""
        return FINISH_INSTRUCTIONS

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"FinishPolicy({self.mode!r})"


_SUBMIT_COMMAND_RE = re.compile(
    rf"""^\s*(?:/bin/)?echo\s+(?:['"])?{re.escape(SUBMIT_MARKER)}(?:['"])?\s*$"""
)


def is_submit_command(cmd: str) -> bool:
    """True if ``cmd`` is the standalone submit command (``echo <marker>``)."""
    return _SUBMIT_COMMAND_RE.match(cmd or "") is not None
