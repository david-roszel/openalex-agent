"""Authentic preserved metadata + explicitly scripted inference; no paid model.

This exercises the submission pipeline, graph, CLM and citation validator. It
does not evaluate a model's research ability and is not a hosted benchmark.
"""
import json
from pathlib import Path
import sys

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openalex_agent.config import Settings
from openalex_agent.corpus import atomic_json
from openalex_agent.live_profiles import thorough_config
from openalex_agent.live_research import source_records
from openalex_agent.research import start_research


def main():
    calls, edited, recovered = [], [False], [False]
    def handle(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        calls.append(payload)
        if "evidence" in payload:
            records = [p for p in payload["evidence"] if len(p.get("abstract_excerpt", "")) >= 20][:2]
            claims = [{"text": "MOCKED ENGINEERING OUTPUT: the recorded abstract states: " + p["abstract_excerpt"][:160],
                "citations": [{"work_id": p["id"], "quote": p["abstract_excerpt"][:160]}]} for p in records]
            value = {"answer": claims, "foundations": [], "comparison": [], "conclusion": [],
                "limitations": ["This is scripted offline validation, not a generated scientific synthesis or an empirical model result."]}
        elif "protected" not in payload:
            value = {"initial_query": "citation networks literature", "components": [
                {"id": "C1", "question": "Source evidence", "search_query": "citation networks literature"}]}
        elif not edited[0]:
            edited[0] = True
            records = list(source_records(payload).values())
            value = {"context_edit": {"path": "active_context.txt", "op": "replace",
                "text": "OFFLINE_SCRIPT_RETAINED_EVIDENCE\n" + json.dumps({"top_papers": records})}}
        elif not recovered[0]:
            recovered[0] = True
            value = {"read_observation": {"index": 0, "start": 0, "chars": 128}}
        else:
            records = [p for p in source_records(payload).values() if len(p.get("abstract_excerpt", "")) >= 20]
            value = {"decision": {"action": "STOP_RETRIEVAL", "arguments": {}, "reason": "Offline scripted evidence exercise ends"},
                "research_status": {"sufficient": True, "gaps": [], "next_query": "",
                    "evidence": [{"component": "C1", "work_id": p["id"], "quote": p["abstract_excerpt"][:100]} for p in records[:2]]}}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}})
    settings = Settings(llm_base_url="https://offline.example/v1/", llm_api_key="OFFLINE_TEST_PLACEHOLDER",
        llm_model="MOCKED_NO_INFERENCE", llm_input_price=1, llm_output_price=5, llm_max_tokens=1000, llm_timeout=2)
    config = thorough_config("MOCKED ENGINEERING EXERCISE: How can citation networks support literature discovery?",
        mode="snapshot", corpus=str(ROOT / "tests/fixtures/development-corpus"))
    job = start_research(config, settings, root=ROOT / "data/submission-validation/mocked-e2e",
        planner_transport=httpx.MockTransport(handle))
    job.join(30)
    state = job.snapshot(); result = state.get("result") or {}
    report = {"evidence_mode": "MOCKED_GENERATION_WITH_AUTHENTIC_HISTORICAL_METADATA",
        "paid_model_requests": 0, "new_live_OpenAlex_requests": 0, "synthetic_token_usage": True,
        "status": state["status"], "synthesis_status": (result.get("synthesis") or {}).get("status"),
        "mechanisms": result.get("mechanisms"), "scripted_calls": len(calls),
        "sources_sent": (result.get("live_accounting") or {}).get("sources_sent_to_model"),
        "source_ids": [p["id"] for p in (result.get("synthesis") or {}).get("sources", [])],
        "independent_scientific_accuracy": "NOT EVALUATED", "research_directory": str(job.directory), "error": state.get("error")}
    atomic_json(ROOT / "docs/submission/offline-mechanism-validation.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["synthesis_status"] == "qualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
