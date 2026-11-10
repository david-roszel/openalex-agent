"""Optional bounded hosted synchronization check; no live retrieval or quality study."""

import json
from pathlib import Path
import uuid

from .actions import make_action
from .config import Settings
from .context_memory import serialized
from .context_planner import ContextPlanner
from .controlled import ControlledController
from .controller import Budget
from .corpus import LocalCorpus, atomic_json
from .llm import HostedJSONPlanner
from .storage import Database


def main():
    settings = Settings.from_env()
    output = Path("data/clm-development-validation/hosted") / str(uuid.uuid4())
    output.mkdir(parents=True)
    corpus = LocalCorpus("data/frozen/three-question-pilot-v1/evidence/graph-ablation-pilot-2026-11-02/corpus")
    question = "citation network scholarly recommendation"
    budget = Budget(max_actions=2, max_rounds=2, max_inspected=4, max_llm_calls=4,
                    max_api_requests=1, page_size=2, max_context_chars=4000, output_k=10)
    with Database(str(output / "metadata.sqlite")) as db, HostedJSONPlanner(settings) as base:
        planner = ContextPlanner(base, mode="clm")
        c = ControlledController(question, corpus, db, architecture="clm", planner=planner, budget=budget,
                                 output=str(output / "runs"))
        c._execute(make_action("SEARCH_PAPERS", question))
        protected, body = planner._parts(c.context())
        overhead = len(serialized(planner._payload(protected, body, active="")))
        pressure = ("SYNTHETIC low-priority historical log. " * 200)[:4000-overhead-200]
        planner.store.transition(pressure, source="environment", operation={"synthetic_pressure_fixture": True})
        result = c.run()
        model_edits = [r for r in planner.store.edits if r["source"] == "model" and r["accepted"]]
        inputs = [json.loads(p.read_text()) for p in sorted(planner.store.directory.glob("request-*.json"))]
        synchronized = any(record["input"].get("active_context", "").startswith(edit["after"])
                           for edit in model_edits for record in inputs
                           if record["call"] > edit["usage"].get("call_index", record["call"]))
        events = db.events(c.run_id)
        report = {"kind": "HOSTED MECHANISM SMOKE WITH SYNTHETIC CONTEXT PRESSURE; NOT EFFECTIVENESS",
            "run_directory": result["directory"], "model": result["model"], "model_configuration": base.configuration,
            "llm_calls": result["llm_usage"]["calls"], "accepted_model_edits": len(model_edits),
            "edited_context_in_subsequent_request": bool(model_edits and synchronized),
            "input_tokens": result["llm_usage"]["input_tokens"], "output_tokens": result["llm_usage"]["output_tokens"],
            "api_requests": result["api_usage"]["requests"], "provider_failures": result["llm_usage"]["failures"],
            "fallbacks": sum(e["kind"] == "invalid_decision" for e in events), "seconds": result["seconds"],
            "stop_reason": result["stop_reason"]}
        atomic_json(output / "validation.json", report)
        print(json.dumps(report, indent=2))
        if not report["edited_context_in_subsequent_request"] or report["provider_failures"] or report["fallbacks"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
