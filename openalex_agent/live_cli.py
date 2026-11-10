"""Submission CLI using exactly the UI's research adapter and Thorough profile."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

from .corpus import atomic_json
from .live_profiles import load_live_settings, thorough_config
from .live_research import answer_markdown
from .live_safety import scrub
from .research import PROJECT_ROOT, progress, start_research


def main(argv=None, *, root=PROJECT_ROOT, settings=None, openalex_transport=None, planner_transport=None):
    parser = argparse.ArgumentParser(description="OpenAlex Live Research: bounded scholarly investigation and evidence-grounded synthesis")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--variant", choices=("combined", "conventional", "graph", "clm", "standard"), default="combined")
    parser.add_argument("--planner", choices=("hosted", "deterministic"), default="hosted")
    parser.add_argument("--snapshot", type=Path, help="Optional small local corpus; live OpenAlex is the default")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--allow-paid", action="store_true", help="Authorize this bounded individual hosted run only")
    parser.add_argument("--ceiling-usd", type=float, default=1.50)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--dry-run", action="store_true", help="Print the effective non-secret configuration; no network")
    parser.add_argument("--check-openalex", action="store_true", help="One real OpenAlex request, no hosted inference")
    parser.add_argument("--output", type=Path, help="Copy the answer Markdown here")
    args = parser.parse_args(argv)
    selected = settings
    try:
        selected = selected or load_live_settings(args.env_file)
        if args.check_openalex:
            return check_openalex(selected, root=Path(root), transport=openalex_transport)
        if not args.question or not args.question.strip():
            parser.error("Enter an academic research question")
        if any(secret and secret in args.question for secret in (selected.api_key, selected.llm_api_key, selected.email)):
            raise ValueError("Remove configured credentials or contact details from the question")
        config = thorough_config(args.question, variant=args.variant, planner=args.planner,
            mode="snapshot" if args.snapshot else "live", corpus=str(args.snapshot) if args.snapshot else None,
            max_seconds=args.max_seconds)
        if args.planner == "hosted" and args.variant != "standard":
            config = replace(config, inference_limits=replace(config.inference_limits, max_llm_cost_usd=args.ceiling_usd))
        if args.dry_run:
            print(json.dumps({"configuration": asdict(config), "model": selected.llm_model,
                "input_upper_bound_per_million": selected.llm_input_price,
                "output_price_per_million": selected.llm_output_price, "paid_requests": 0}, indent=2))
            return 0
        if config.planner == "hosted" and not args.allow_paid:
            raise ValueError("Hosted inference was not authorized. Review --dry-run, then pass --allow-paid for this individual bounded run.")
        job = start_research(config, selected, root=Path(root), openalex_transport=openalex_transport, planner_transport=planner_transport)
        last = -1
        try:
            while True:
                state = job.snapshot()
                if len(state["events"]) != last:
                    last = len(state["events"])
                    p = progress(state)
                    print(json.dumps({"status": state["status"], "phase": state.get("phase"),
                        "actions": p["actions"], "records": p["inspected"], "model_calls": p["planner_calls"]}), file=sys.stderr, flush=True)
                if state["status"] not in ("starting", "running"):
                    break
                job.join(0.25)
        except KeyboardInterrupt:
            # The worker's paid hold/checkpoints survive. Do not silently retry
            # an in-flight generation on a later invocation.
            print("Interrupted. Research artifacts and any spending hold are retained at " + str(job.directory), file=sys.stderr)
            return 130
        job.join(5)
        state = job.snapshot()
        result = state.get("result")
        print("Artifacts: " + str(job.directory), file=sys.stderr)
        if result:
            text = answer_markdown(result)
            print(text)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text, encoding="utf-8")
        if state.get("error"):
            print(state["error"]["message"], file=sys.stderr)
            return 1
        if config.planner == "deterministic" or args.variant == "standard":
            return 0
        synthesis = (result or {}).get("synthesis") or {}
        return 0 if synthesis.get("status") == "qualified" and synthesis.get("question_completion") != "incomplete_evidence" else 3
    except (ValueError, OSError) as exc:
        secrets = (selected.api_key, selected.llm_api_key, selected.email) if selected else ()
        print(scrub(str(exc), secrets), file=sys.stderr)
        return 2


def check_openalex(settings, *, root, transport=None):
    from .openalex import OpenAlexClient, RetrievalError
    from .storage import Database
    directory = root / "data/submission-validation"
    directory.mkdir(parents=True, exist_ok=True)
    record = {"kind": "LIVE_OPENALEX_CONNECTIVITY_ONLY", "hosted_inference_calls": 0}
    selected = replace(settings, max_requests=1, max_retries=0, timeout=min(settings.timeout, 15))
    with Database(str(directory / "connectivity.sqlite")) as db, OpenAlexClient(db, selected) as client:
        try:
            raw = client._request("/works", {"search": "retrieval augmented generation knowledge intensive NLP", "per-page": 2}, refresh=True)
            papers = client.normalize(raw["results"])
            record.update(status="success", papers=[p.to_dict() for p in papers], usage=asdict(client.usage))
        except RetrievalError:
            from .research import retrieval_error_message
            record.update(status="failed", message=retrieval_error_message(db.events(client.run_id)),
                          usage=asdict(client.usage), errors=db.events(client.run_id))
    atomic_json(directory / "openalex-connectivity.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0 if record["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
