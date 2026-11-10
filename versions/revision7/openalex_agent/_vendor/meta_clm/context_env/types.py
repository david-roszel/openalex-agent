# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The value ``ContextEnv.step`` returns to the agent loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StepResult:
    """One ``ContextEnv.step``; the agent loop appends ``render()`` as the tool result."""

    result: Any
    ctx_changed: bool
    stdout_block: str
    readout: str
    notes: str
    exec_time: float
    touched_ctx: bool

    def render(self) -> str:
        """Concatenate stdout, readout, then notes."""
        return self.stdout_block + self.readout + self.notes
