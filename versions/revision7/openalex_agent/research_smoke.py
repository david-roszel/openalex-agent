"""Bounded real-API validation through the same adapter as the Streamlit app."""

import json
from dataclasses import replace

from .config import Settings
from .controller import Budget
from .corpus import atomic_json
from .research import ResearchConfig, progress, start_research


def main():
    settings = Settings.from_env()
    if not settings.api_key:
        raise SystemExit("Live smoke test skipped: OPENALEX_API_KEY is not configured")
    # One deliberate planner decision at most; no retries or provider switching.
    settings = replace(settings, max_requests=min(settings.max_requests, 3), max_retries=0,
                       max_cost_usd=min(settings.max_cost_usd, 0.01))
    planner = "hosted" if all((settings.llm_base_url, settings.llm_api_key, settings.llm_model)) else "deterministic"
    config = ResearchConfig("How do quantum error correction methods improve fault tolerance in superconducting qubits?",
        planner=planner, budget=Budget(max_actions=2, max_rounds=2, max_inspected=6,
            max_api_requests=3, max_llm_calls=1, page_size=3, output_k=6))
    job = start_research(config, settings)
    previous = -1
    while True:
        state = job.snapshot()
        if len(state["events"]) != previous:
            previous = len(state["events"])
            print(json.dumps({"status": state["status"], "recorded_events": previous, **progress(state)}), flush=True)
        if state["status"] not in ("starting", "running"):
            break
        job.join(0.5)
    job.join(5)
    state = job.snapshot()
    result = state.get("result") or {}
    report = {"kind": "LIVE RETRIEVAL ENGINEERING SMOKE; NOT RELEVANCE EVALUATION", "status": state["status"],
        "question": config.question, "directory": str(job.directory), "run_id": state["run_id"],
        "progress": progress(state), "comparison": state["comparison"], "error": state["error"],
        "api_usage": result.get("api_usage"), "llm_usage": result.get("llm_usage"),
        "model": result.get("model"), "seconds": result.get("seconds"),
        "ranked_titles": [item["paper"]["title"] for item in result.get("ranked", [])]}
    atomic_json(job.directory / "live-validation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if state["status"] != "completed" or not report["progress"]["successful_responses"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
