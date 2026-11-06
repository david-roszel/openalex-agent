import httpx
import pytest
from openalex_agent.controller import Budget, Controller
from openalex_agent.openalex import OpenAlexClient


def test_interrupted_action_replays_from_checkpoint_without_double_charging(db, settings, raw_work, tmp_path):
    def interrupt(request):
        raise KeyboardInterrupt()
    client = OpenAlexClient(db, settings, transport=httpx.MockTransport(interrupt))
    first = Controller("retrieval", client, db, output=str(tmp_path), budget=Budget(max_actions=1))
    with pytest.raises(KeyboardInterrupt):
        first.run()
    assert first.actions == 1 and first.pending
    client.close()
    def success(request):
        return httpx.Response(200, json={"results": [raw_work], "meta": {"next_cursor": None}})
    client = OpenAlexClient(db, settings, transport=httpx.MockTransport(success))
    resumed = Controller("retrieval", client, db, output=str(tmp_path), budget=Budget(max_actions=1), resume=str(first.directory))
    result = resumed.run()
    assert result["actions"] == 1 and len(result["inspected_ids"]) == 1
    assert len(resumed.history) == 1
    assert result["api_usage"]["requests"] == 2  # interrupted attempt still counts
    client.close()
