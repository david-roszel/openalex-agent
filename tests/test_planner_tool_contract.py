"""Offline regressions for the frozen pilot's two rejection types, not LLM results."""

from dataclasses import asdict
import json

import pytest

from openalex_agent.actions import Action, CONTROLLED_TOOL_CONTRACT, RELIABILITY_VERSION, make_action
from openalex_agent.context_planner import ContextPlanner, EngineeringContextModel, InferenceLimits, CONTEXT_POLICY_VERSION
from openalex_agent.controlled import ARCHITECTURES, GRAPH_FIELDS, ControlledController
from openalex_agent.controlled_diagnostics import SeededCorpus
from openalex_agent.controller import Budget
from openalex_agent.corpus import freeze
from openalex_agent.experiments import run_experiments
from openalex_agent.models import Paper


def corpus(tmp_path):
    path = tmp_path / "corpus"
    papers = [Paper.from_openalex({"id": f"W{i}", "title": "SYNTHETIC retrieval evidence",
                "referenced_works": [f"W{i+1}"] if i < 3 else []}) for i in range(1, 4)]
    freeze(papers, path, acquisition=[{"source": "SYNTHETIC REGRESSION ONLY"}])
    return path


class Sequence(EngineeringContextModel):
    def __init__(self, actions):
        super().__init__()
        self.actions = actions
        self.inputs = []

    def _call(self, system, payload):
        self.inputs.append((system, payload))
        return self.respond(system, payload, {"decision": self.actions[self.usage.calls]})


@pytest.mark.parametrize("architecture", ARCHITECTURES[1:])
@pytest.mark.parametrize("failure", ["loaded_inspection", "completed_expansion"])
def test_original_invalid_actions_remain_invalid_and_fallbacks_keep_their_cost(tmp_path, db, architecture, failure):
    path = corpus(tmp_path)
    action = make_action("INSPECT_PAPER" if failure == "loaded_inspection" else "EXPAND_REFERENCES", "W1")
    # Reproduce canonical/short aliases used by the real pilot; both must hit
    # the same eligibility/completed-action guard.
    alias = {**action.to_dict(), "arguments": {"paper_id": "W1"}}
    model = Sequence([action.to_dict(), alias])
    limits = InferenceLimits(max_total_tokens=100000, max_active_tokens=4096, max_input_tokens=8192)
    planner = ContextPlanner(model, mode="clm" if architecture.startswith("clm") else "conventional", limits=limits)
    budget = Budget(max_actions=3, max_rounds=10, max_llm_calls=20, page_size=1, patience=10)
    controller = ControlledController("retrieval", SeededCorpus(path, [1]), db,
        architecture=architecture, planner=planner, budget=budget, output=str(tmp_path / "runs"))
    result = controller.run()
    events = [json.loads(line) for line in (controller.directory / "events.jsonl").read_text().splitlines()]
    invalid = [e for e in events if e["kind"] == "invalid_decision"]
    expected = 2 if failure == "loaded_inspection" else 1
    assert len(invalid) == expected
    rule = ("Target fails observed metadata, relevance or common path-depth guard"
            if failure == "loaded_inspection" else "Repeated action provides no new information")
    assert all(e["validation_rule"] == rule and e["error_code"] == "invalid_action" for e in invalid)
    # A fallback still executes a charged acquisition action. No free retry,
    # duplicate success, hidden extra model call, or softened scoring rule.
    assert result["actions"] == len(controller.history) == 3
    assert planner.usage.calls == len(model.inputs) == 2
    assert len(controller.completed) == 3
    assert sum(a["action"] == action.action and a["arguments"] == action.arguments for a in controller.history) == (failure == "completed_expansion")
    assert result["budget"] == asdict(budget)
    assert result["inference_limits"] == asdict(limits)
    assert result["experimental_implementation_version"] == CONTEXT_POLICY_VERSION
    assert planner.configuration["context_policy"] == CONTEXT_POLICY_VERSION
    assert planner.configuration["reliability_version"] == RELIABILITY_VERSION
    assert all(CONTROLLED_TOOL_CONTRACT in system for system, _ in model.inputs)
    first = model.inputs[0][1]
    proposals = first["protected"]["proposals"]
    assert make_action("INSPECT_PAPER", "W1").signature not in {Action.parse(p).signature for p in proposals}
    assert any(p["action"] == "INSPECT_PAPER" and p["arguments"]["paper_id"].endswith("/W2") for p in proposals)
    if failure == "completed_expansion":
        assert action.signature not in {Action.parse(p).signature for p in model.inputs[1][1]["protected"]["proposals"]}
    assert all(r["study_tokens"]["full_input_tokens"] <= 8192 and
               r["study_tokens"]["active_tokens"] + r["study_tokens"]["recovery_tokens"] <= 4096
               for p in planner.store.directory.glob("request-*.json") for r in [json.loads(p.read_text())])
    assert result["api_usage"]["requests"] == 0


def test_all_five_configurations_record_new_version_and_keep_matched_tools_and_context_interventions(tmp_path, db):
    path = corpus(tmp_path)
    question = tmp_path / "questions.json"
    question.write_text(json.dumps([{"id": "SYNTHETIC", "question": "retrieval"}]))
    output = tmp_path / "experiment"
    budget = Budget(max_actions=10, max_rounds=10, max_llm_calls=20, page_size=1)
    limits = InferenceLimits(max_total_tokens=100000, max_active_tokens=4096, max_input_tokens=8192)
    rows = run_experiments(str(path), str(question), db,
        lambda: Sequence([make_action("STOP_RETRIEVAL").to_dict()]), output=str(output),
        suite="controlled-five", action_budgets=(10,), trials=1, budget=budget,
        inference_limits=limits, allow_deterministic=True)
    assert {r["method"] for r in rows} == set(ARCHITECTURES)
    manifest = json.loads((output / "experiment.json").read_text())
    assert manifest["experimental_implementation_version"] == RELIABILITY_VERSION
    assert manifest["base_budget"] == asdict(budget) and manifest["inference_limits"] == asdict(limits)
    common, systems = [], []
    for row in rows:
        directory = output / "runs" / row["run_id"]
        requests = sorted((directory / "context").glob("request-*.json"))
        if row["method"] == "standard":
            assert not requests and row["llm_calls"] == 0 and row["actions"] == 1
            continue
        request = json.loads(requests[0].read_text())
        payload = request["input"]
        protected = {k:v for k,v in payload["protected"].items() if k != "context_management"}
        common.append(protected)
        systems.append(request["system"].split(CONTROLLED_TOOL_CONTRACT)[0])
        is_clm, is_graph = row["method"].startswith("clm"), row["method"].endswith("_graph")
        assert ("active_context" in payload) == is_clm
        assert ("observation" in payload) != is_clm
        body = (json.loads(payload["active_context"].strip())["changes"] if is_clm else payload["observation"])
        assert bool(GRAPH_FIELDS.intersection(body)) == is_graph
        assert (directory / "graph.json").exists() == is_graph
        assert CONTROLLED_TOOL_CONTRACT in request["system"]
    assert all(value == common[0] for value in common)
    # The shared prefix/schema/contract is identical; only CLM edit instructions
    # below it differ, as they did before this revision.
    assert all(value == systems[0] for value in systems)
