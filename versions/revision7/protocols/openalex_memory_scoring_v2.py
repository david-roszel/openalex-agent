"""Deterministic reference scorer for OpenAlex memory V2 final probes.

No hosted model calls, file I/O, or scientific judgments. Use only on the
private answer key and recorded branch response. All six items are binary;
primary uses q1..q5 only. Validate inputs separately against the supplied
JSON schemas and provenance/exposure invariants before relying on scores.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

QUESTION_IDS = ("q1", "q2", "q3", "q4", "q5", "q6")


def _is_work_id(value: Any) -> bool:
    return (isinstance(value, str) and value.startswith("W")
            and value[1:].isdigit() and len(value) > 1)


def _valid_answer(answer: Any) -> bool:
    """Exact response contract; extra keys and malformed values are invalid."""
    if not isinstance(answer, Mapping) or set(answer) != set(QUESTION_IDS):
        return False
    def obj(name: str, keys: set[str]) -> bool:
        return isinstance(answer[name], Mapping) and set(answer[name]) == keys
    if not obj("q1", {"work_id"}) or not obj("q3", {"work_id"}):
        return False
    if not all(_is_work_id(answer[q]["work_id"]) for q in ("q1", "q3")):
        return False
    if not obj("q2", {"year"}):
        return False
    y = answer["q2"]["year"]
    if type(y) is not int or not 1 <= y <= 2100:
        return False
    if not obj("q4", {"citing_work_id", "cited_work_id"}):
        return False
    if not all(_is_work_id(answer["q4"][k]) for k in ("citing_work_id", "cited_work_id")):
        return False
    if not obj("q5", {"path"}):
        return False
    path = answer["q5"]["path"]
    if not isinstance(path, list) or len(path) != 3 or not all(_is_work_id(v) for v in path):
        return False
    if not obj("q6", {"status"}):
        return False
    return answer["q6"]["status"] in ("SUPPORTED", "NOT_ESTABLISHED")


def score_response(response: str | Mapping[str, Any] | None,
                   gold: Mapping[str, Any], *, terminal_failure: bool = False) -> dict[str, Any]:
    """Score one A/B branch. Gold must be a valid, privately held answer object.

    Malformed entire JSON scores all zero for the branch; don't infer missing
    facts, forgive reversed citation direction, or invoke another LLM.
    """
    if not _valid_answer(gold):
        raise ValueError("Invalid gold key; do not score an unverified case")
    parsed: Any = None
    if not terminal_failure:
        try:
            parsed = json.loads(response) if isinstance(response, str) else response
        except (json.JSONDecodeError, TypeError):
            parsed = None
    valid = _valid_answer(parsed)
    correctness = {q: int(valid and parsed[q] == gold[q]) for q in QUESTION_IDS}
    primary = sum(correctness[q] for q in QUESTION_IDS[:5]) / 5
    return {
        "valid_final_response": valid,
        "terminal_failure": bool(terminal_failure),
        "items": correctness,
        "primary_score": primary,
        "source_identity_year_score": sum(correctness[q] for q in ("q1", "q2", "q3")) / 3,
        "citation_relation_score": (correctness["q4"] + correctness["q5"]) / 2,
        "evidence_calibration_score": correctness["q6"],
    }


def factorial_effects(scores: Mapping[str, float]) -> dict[str, float]:
    """One independent matched case; scores already aggregate any repetitions."""
    y00 = scores["llm"]
    y10 = scores["llm_graph"]
    y01 = scores["clm"]
    y11 = scores["clm_graph"]
    return {
        "graph": ((y10 - y00) + (y11 - y01)) / 2,
        "clm": ((y01 - y00) + (y11 - y10)) / 2,
        "interaction": y11 - y10 - y01 + y00,
        "graph_alone_vs_llm": y10 - y00,
        "clm_alone_vs_llm": y01 - y00,
        "combined_vs_graph": y11 - y10,
        "combined_vs_clm": y11 - y01,
    }


if __name__ == "__main__":
    import unittest

    class ScoringTests(unittest.TestCase):
        def setUp(self) -> None:
            # Synthetic IDs are TEST FIXTURES, never actual OpenAlex claims.
            self.key = {
                "q1": {"work_id": "W100"},
                "q2": {"year": 2020},
                "q3": {"work_id": "W200"},
                "q4": {"citing_work_id": "W200", "cited_work_id": "W100"},
                "q5": {"path": ["W300", "W200", "W100"]},
                "q6": {"status": "NOT_ESTABLISHED"},
            }

        def test_perfect(self) -> None:
            self.assertEqual(score_response(self.key, self.key)["primary_score"], 1.0)

        def test_reversed_citation_and_wrong_path(self) -> None:
            answer = json.loads(json.dumps(self.key))
            answer["q4"] = {"citing_work_id": "W100", "cited_work_id": "W200"}
            answer["q5"]["path"].reverse()
            result = score_response(answer, self.key)
            self.assertEqual(result["primary_score"], 0.6)
            self.assertEqual(result["citation_relation_score"], 0)

        def test_unknown_relation(self) -> None:
            answer = json.loads(json.dumps(self.key))
            answer["q6"]["status"] = "SUPPORTED"
            result = score_response(answer, self.key)
            self.assertEqual(result["primary_score"], 1.0)
            self.assertEqual(result["evidence_calibration_score"], 0)

        def test_failure_and_malformed(self) -> None:
            self.assertEqual(score_response(None, self.key, terminal_failure=True)["primary_score"], 0)
            self.assertFalse(score_response("not-json", self.key)["valid_final_response"])

        def test_factorial(self) -> None:
            effects = factorial_effects({"llm": .2, "llm_graph": .4, "clm": .3, "clm_graph": .6})
            self.assertAlmostEqual(effects["graph"], .25)
            self.assertAlmostEqual(effects["clm"], .15)
            self.assertAlmostEqual(effects["interaction"], .1)

    unittest.main(verbosity=2)
