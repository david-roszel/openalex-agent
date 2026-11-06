import json

import pandas as pd
import pytest

from openalex_agent.context_planner import InferenceLimits
from openalex_agent.controlled import ARCHITECTURES
from openalex_agent.controlled_diagnostics import fixture
from openalex_agent.controller import Budget
from openalex_agent.corpus import freeze
from openalex_agent.experiments import build_pool, run_experiments
from openalex_agent.llm import DeterministicPlanner
from openalex_agent.research_metrics import paired_comparisons, retrieval_diagnostics


def test_five_methods_three_trials_and_deterministic_replay_preserve_legacy(db, tmp_path):
    question, papers, _, _, _ = fixture("G1")
    corpus = tmp_path / "corpus"
    freeze(papers, corpus, acquisition=[{"source": "SYNTHETIC FIVE METHOD ENGINEERING ONLY"}])
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{"id": "q1", "question": question}]))
    output = tmp_path / "five"
    rows = run_experiments(str(corpus), str(questions), db, DeterministicPlanner, output=str(output),
        suite="controlled-five", allow_deterministic=True, action_budgets=(3,),
        budget=Budget(page_size=1, max_llm_calls=6, max_rounds=3))
    assert len(rows) == 15 and {r["method"] for r in rows} == set(ARCHITECTURES)
    assert all(r["deterministic_smoke"] for r in rows)
    assert {r["trial"] for r in rows} == {0, 1, 2}
    assert all(r["api_requests"] == 0 for r in rows)
    assert all(r["end_to_end_seconds"] >= r["seconds"] for r in rows)
    assert all(r["context_edits"] > 0 for r in rows if r["method"].startswith("clm"))
    assert all(r["context_edits"] == 0 for r in rows if not r["method"].startswith("clm"))
    assert build_pool(str(output), str(tmp_path / "pool.csv")) > 0
    assert "method" not in pd.read_csv(tmp_path / "pool.csv").columns
    for method in ARCHITECTURES:
        outcomes = []
        for row in rows:
            if row["method"] == method:
                from pathlib import Path
                result = json.loads((Path(row["directory"]) / "result.json").read_text())
                state = json.loads((Path(row["directory"]) / "state.json").read_text())
                outcomes.append(([r["paper"]["id"] for r in result["ranked"]], state["history"]))
        assert all(outcome == outcomes[0] for outcome in outcomes)
    manifest = json.loads((output / "experiment.json").read_text())
    assert manifest["suite"] == "controlled-five" and manifest["status"] == "complete"


def test_paired_statistics_average_trials_before_question_bootstrap_and_allow_regressions():
    rows = []
    values = {"llm": 1, "llm_graph": 0, "clm": 2, "clm_graph": 1}
    for question in ("q1", "q2"):
        for method, value in values.items():
            for trial in range(3):
                rows.append({"query_id": question, "method": method, "trial": trial, "action_budget": 5, "score": value})
    paired = paired_comparisons(rows, "score")
    graph = next(r for r in paired if r["left"] == "llm_graph" and r["right"] == "llm")
    assert graph["questions"] == 2 and graph["mean_difference"] == -1 and graph["bootstrap_95_ci"] == [-1, -1]
    assert next(r for r in paired if r["left"] == "interaction")["mean_difference"] == 0
    assert paired == paired_comparisons(rows, "score")


def test_independent_subtopic_labels_are_separate_from_relevance_and_planner():
    result = retrieval_diagnostics(["a", "a", "b", "unjudged"], {"a": 3, "b": 0}, 4,
                                  subtopics={"a": ["s1"]}, expected_subtopics=["s1", "s2"])
    assert result["unique_relevant_papers"] == 1 and result["actions_per_relevant_paper"] == 4
    assert result["subtopic_coverage"] == .5


def test_research_token_budget_stops_without_spending_remaining_actions_on_fallback(db, tmp_path):
    q, papers, _, _, _ = fixture("G1")
    corpus = tmp_path / "corpus"
    freeze(papers, corpus, acquisition=[{"source": "SYNTHETIC BUDGET TEST ONLY"}])
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{"id": "q1", "question": q}]))
    rows = run_experiments(str(corpus), str(questions), db, DeterministicPlanner, output=str(tmp_path / "experiment"),
        suite="controlled-five", methods=("clm",), allow_deterministic=True, trials=1, action_budgets=(3,),
        budget=Budget(page_size=1), inference_limits=InferenceLimits(max_total_tokens=1))
    assert rows[0]["llm_calls"] == 0 and rows[0]["fallbacks"] == 0
    assert rows[0]["actions"] == 1 and rows[0]["stop_reason"] == "budget_exhausted"
    assert any(e["kind"] == "context_budget_stop" and e["code"] == "token_budget_exhausted" for e in db.events(rows[0]["run_id"]))
