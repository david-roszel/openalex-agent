"""Synthetic API records strictly for offline tests; these are NOT real papers."""

import pytest
import json
from pathlib import Path
from openalex_agent.config import Settings
from openalex_agent.storage import Database


@pytest.fixture
def db():
    with Database(":memory:") as database:
        yield database


@pytest.fixture
def settings():
    return Settings(min_interval=0, max_cost_usd=1)


@pytest.fixture
def raw_work():
    path = Path(__file__).parent / "fixtures" / "openalex-list.synthetic.json"
    return json.loads(path.read_text(encoding="utf-8"))["results"][0]
