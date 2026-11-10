# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Sandbox paths for the live-context mirror.

Kept as a tiny namespace object so callers can rebind
``clm_harness.clm_agent.harness._CTX_DIR`` (etc.) BEFORE constructing the agent
and every lookup here still sees the new values. A copied string at import time
would freeze the default ``/tmp/.live_ctx`` and break concurrent shards.

``Paths`` is a view of that module's globals, not a snapshot.
"""

from __future__ import annotations

from typing import Any


# Defaults — also the values ``clm_agent.harness`` starts with.
CTX_DIR = "/tmp/.live_ctx"
CTX_FILE = f"{CTX_DIR}/LIVE_CTX_MAIN.txt"
STATE_DIR = "/tmp/.bash_ctx_state"


class Paths:
    """Late-bound view of a harness module's ``_CTX_DIR`` / ``_CTX_FILE`` / ``_STATE_DIR``."""

    def __init__(self, ns: Any) -> None:
        self._ns = ns

    def _get(self, name: str, default: str) -> str:
        return str(getattr(self._ns, name, default))

    @property
    def ctx_dir(self) -> str:
        return self._get("_CTX_DIR", CTX_DIR)

    @property
    def ctx_file(self) -> str:
        return self._get("_CTX_FILE", CTX_FILE)

    @property
    def state_dir(self) -> str:
        return self._get("_STATE_DIR", STATE_DIR)
