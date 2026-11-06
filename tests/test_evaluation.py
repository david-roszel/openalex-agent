import json
import math
import pytest
from openalex_agent.evaluation import import_judgments, load_judgments, metrics
from openalex_agent.experiments import build_pool, report, run_experiments, score_experiment
from openalex_agent.controller import Budget
from openalex_agent.llm import DeterministicPlanner
from openalex_agent.corpus import freeze
from openalex_agent.models import Paper


def test_metrics_known_example_and_graded_ndcg():
    labels = {"a": 3, "b": 1, "c": 0}
    values = metrics(["b", "c", "a"], labels, k=2, scope="exhaustive", inspected=["a", "c"], actions=2)
    assert values["precision_at_k"] == values["recall_at_k"] == 0.5
    assert values["ndcg_at_k"] == pytest.approx(1 / (7 + 1/math.log2(3)))
    assert values["judged_irrelevant_inspection_fraction"] == 0.5
    assert values["known_relevant_per_action"] == 0.5


def test_unjudged_are_unknown_not_irrelevant():
    values = metrics(["a", "unknown"], {"a": 1, "b": 1}, k=2)
    assert values["precision_at_k"] is None and values["ndcg_at_k"] is None
    assert values["precision_lower_bound"] == 0.5
    assert values["precision_upper_bound"] == 1
    assert values["judged_precision"] == 1
    assert values["recall_at_k"] == 0.5 and values["recall_scope"] == "judged_pool"
    assert metrics([], {}, k=10)["recall_at_k"] is None
    with pytest.raises(ValueError):
        metrics(["a", "a"], {}, k=2)


def test_human_judgment_import_and_conflict(tmp_path):
    csv = tmp_path / "judgments.csv"
    output = tmp_path / "labels.json"
    csv.write_text("query_id,paper_id,grade,assessor\nq1,W1,2,reviewer-a\n")
    import_judgments(str(csv), str(output))
    assert load_judgments(str(output)) == ({"q1": {"https://openalex.org/W1": 2}}, "pooled")
    with csv.open("a") as stream:
        stream.write("q1,W1,0,reviewer-b\n")
    with pytest.raises(ValueError, match="Conflicting"):
        import_judgments(str(csv), str(output))


def test_four_methods_experiment_and_pool_without_fabricated_quality(db, raw_work, tmp_path):
    corpus = tmp_path / "corpus"
    freeze([Paper.from_openalex(raw_work)], corpus, acquisition=[{"source": "SYNTHETIC TEST ONLY"}])
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{"id": "q1", "question": "adaptive retrieval"}]))
    output = tmp_path / "experiment"
    with pytest.raises(ValueError, match="LLM comparisons require"):
        run_experiments(str(corpus), str(questions), db, DeterministicPlanner, output=str(output))
    rows = run_experiments(str(corpus), str(questions), db, DeterministicPlanner, output=str(output),
                           action_budgets=(2,), trials=1, budget=Budget(max_inspected=1), allow_deterministic=True)
    assert {r["method"] for r in rows} == {"standard", "reformulated", "iterative", "graph"}
    assert all(r["deterministic_smoke"] and "recall_at_k" not in r for r in rows)
    assert build_pool(str(output), str(tmp_path / "pool.csv")) == 1
    with pytest.raises(ValueError, match="relevance judgments"):
        report(str(output))
    source = tmp_path / "judgments.csv"
    source.write_text("query_id,paper_id,grade,assessor\nq1,W1,2,SYNTHETIC-TEST-ASSESSOR\n")
    labels = tmp_path / "labels.json"
    import_judgments(str(source), str(labels))
    assert score_experiment(str(output), str(labels)) == 4
    import pandas as pd
    scored = pd.read_csv(output / "metrics.csv")
    assert (scored["recall_at_k"] == 1).all()


def test_labeled_report_is_computed_from_recorded_outputs(db, raw_work, tmp_path):
    corpus = tmp_path / "corpus"
    freeze([Paper.from_openalex(raw_work)], corpus, acquisition=[{"source": "SYNTHETIC TEST ONLY"}])
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{"id": "q1", "question": "retrieval"}]))
    source = tmp_path / "judgments.csv"
    source.write_text("query_id,paper_id,grade,assessor\nq1,W1,2,SYNTHETIC-TEST-ASSESSOR\n")
    labels = tmp_path / "labels.json"
    import_judgments(str(source), str(labels))
    output = tmp_path / "experiment"
    rows = run_experiments(str(corpus), str(questions), db, DeterministicPlanner, output=str(output),
        labels_path=str(labels), action_budgets=(1,), budget=Budget(max_inspected=1), allow_deterministic=True)
    assert all(r["recall_at_k"] == 1 for r in rows)
    paths = report(str(output))
    assert len(paths) == 6
    assert all(__import__("pathlib").Path(p).exists() for p in paths)
