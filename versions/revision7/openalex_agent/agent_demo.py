"""Milestone 3 engineering demo: MOCKED hosted LLM and SYNTHETIC OpenAlex data.

No external request, API credential, paid call or research result is involved.
The mock provider uses current compact state to choose the next controlled action.
"""

import argparse
import json
from pathlib import Path
import uuid
import httpx
from .actions import make_action
from .config import Settings
from .controller import Budget, Controller
from .llm import HostedJSONPlanner
from .openalex import OpenAlexClient
from .storage import Database


def run_demo(output: str = "data/milestone3-demo") -> dict:
    directory = Path(output) / str(uuid.uuid4())
    directory.mkdir(parents=True, exist_ok=True)
    contexts = []
    retrieval_requests = []
    def work(number, references=()):
        return {"id": f"https://openalex.org/W{number}", "title": f"SYNTHETIC TEST adaptive retrieval paper {number}",
                "referenced_works": [f"W{i}" for i in references],
                "topics": [{"id": "T1", "display_name": "adaptive retrieval"}],
                "abstract_inverted_index": {"Synthetic": [0], "adaptive": [1], "retrieval": [2]},
                "authorships": [{"author": {"id": "A1", "display_name": "SYNTHETIC AUTHOR"},
                                 "institutions": [{"id": "I1", "display_name": "SYNTHETIC INSTITUTION"}]}]}
    works = {1: work(1, [2]), 2: work(2, [3]), 3: work(3), 4: work(4), 5: work(5, [1]), 6: work(6)}
    def retrieve(request):
        retrieval_requests.append({"path": request.url.path, "params": dict(request.url.params)})
        if request.url.path == "/works/W3":
            return httpx.Response(200, json=works[3])
        filters = request.url.params.get("filter", "")
        if filters == "openalex:W2":
            result = [works[2]]
        elif filters == "topics.id:T1":
            result = [works[6]]
        elif filters == "cites:W1":
            result = [works[5]]
        elif request.url.params.get("search") == "adaptive retrieval followup":
            result = [works[4]]
        elif request.url.params.get("search") == "adaptive retrieval":
            result = [works[1]]
        else:
            raise AssertionError("Unexpected mock retrieval action")
        return httpx.Response(200, json={"results": result, "meta": {"next_cursor": None}})
    def decide(request):
        body = json.loads(request.content)
        state = json.loads(body["messages"][-1]["content"])
        contexts.append(state)
        if state["round"] == 1:
            assert "https://openalex.org/W3" not in json.dumps(state)
        plan = {1: ("EXPAND_REFERENCES", "W1"), 2: ("INSPECT_PAPER", "W3"),
                3: ("SEARCH_PAPERS", "adaptive retrieval followup"), 4: ("EXPLORE_TOPIC", "T1"),
                5: ("EXPAND_CITATIONS", "W1"), 6: ("STOP_RETRIEVAL", None)}
        name, target = plan[state["round"]]
        action = make_action(name, target, "MOCK decision from current observed state")
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(action.to_dict())}, "finish_reason": "stop"}]})
    settings = Settings(min_interval=0, llm_base_url="https://mock-provider.example.invalid/v1",
                        llm_api_key="MOCK-ONLY-KEY", llm_model="mock-json-model")
    with Database(str(directory / "metadata.sqlite")) as db:
        with OpenAlexClient(db, settings, transport=httpx.MockTransport(retrieve)) as client:
            client.experiment_kind = "synthetic_mock"
            client.fingerprint = "synthetic:milestone3-demo"
            with HostedJSONPlanner(settings, transport=httpx.MockTransport(decide)) as planner:
                planner.name = "mock-hosted-json"
                controller = Controller("adaptive retrieval", client, db, planner=planner,
                    budget=Budget(max_actions=10, max_rounds=10, max_inspected=20, output_k=6),
                    output=str(directory / "runs"))
                result = controller.run()
                events = db.events(result["run_id"])
    assert result["stop_reason"] == "planner_stop" and result["actions"] == 6
    assert len(result["inspected_ids"]) == 6 and result["llm_usage"]["calls"] == 6
    assert len(retrieval_requests) == 6 and len(contexts) == 6
    assert {a["action"] for a in controller.history} == {"SEARCH_PAPERS", "EXPAND_REFERENCES", "INSPECT_PAPER", "EXPLORE_TOPIC", "EXPAND_CITATIONS"}
    assert any(e["kind"] == "stop_decision" for e in events)
    assert all(len(json.dumps(c, ensure_ascii=False)) <= controller.budget.max_context_chars for c in contexts)
    report = {"kind": "MOCK HOSTED / SYNTHETIC RETRIEVAL ENGINEERING ACCEPTANCE ONLY",
              "real_external_requests": 0, "run_id": result["run_id"], "directory": result["directory"],
              "retrieval_actions": result["actions"], "mock_provider_calls": len(contexts),
              "mock_retrieval_requests": len(retrieval_requests), "inspected_papers": len(result["inspected_ids"]),
              "checks_passed": True, "stop_reason": result["stop_reason"]}
    (directory / "acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/milestone3-demo")
    print(json.dumps(run_demo(parser.parse_args().output)))
