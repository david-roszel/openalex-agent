"""Run the existing paired pilot with conservative, shared Groq token pacing.

No automatic retries: a failed call still counts as a failure. Waits between
calls are recorded separately from HTTP/model latency. Production planners and
retrieval policies are unchanged.
"""

import argparse
import json
import math
from pathlib import Path
import re
import time

from .config import Settings
from .corpus import atomic_json
from .llm import HostedJSONPlanner, PlannerError
from .models import utcnow
from .pilot import run_pilot, analyze_pilot


def duration_seconds(value):
    if not isinstance(value, str):
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        result = float(value)
    elif re.fullmatch(r"(?:\d+(?:\.\d+)?(?:ms|s|m|h))+", value):
        result = sum(float(n)*{"ms": .001, "s": 1, "m": 60, "h": 3600}[unit]
                     for n, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", value))
    else:
        return None
    return result if math.isfinite(result) else None


class TokenPacer:
    def __init__(self, path, *, clock=time.perf_counter, sleep=time.sleep):
        self.path, self.clock, self.sleep = Path(path), clock, sleep
        self.next_allowed = 0.0

    def wait(self):
        started = self.clock()
        remaining = self.next_allowed-started
        if remaining > 90:
            raise PlannerError("quota_wait_exceeds_pilot_limit")
        while remaining > 0:
            self.sleep(min(remaining, 30))
            remaining = self.next_allowed-self.clock()
        return self.clock()-started

    def observe(self, response):
        headers = {k: v for k, v in response.headers.items()
                   if k.startswith("x-ratelimit-") or k == "retry-after"}
        token_reset = duration_seconds(headers.get("x-ratelimit-reset-tokens"))
        retry = duration_seconds(headers.get("retry-after")) or 0.0
        # Wait for the advertised token bucket to refill, not the daily RPD reset.
        delay = max(1.0, token_reset if token_reset is not None else 60.0, retry)+1.0
        self.next_allowed = self.clock()+delay
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as stream:
            stream.write(json.dumps({"timestamp": utcnow(), "http_status": response.status_code,
                "headers": headers, "next_wait_seconds": delay})+"\n")


class PacedPlanner(HostedJSONPlanner):
    def __init__(self, settings, pacer, **kwargs):
        super().__init__(settings, **kwargs)
        self.pacer, self.quota_wait_seconds = pacer, 0.0
        self.http.event_hooks["response"].append(pacer.observe)

    def _call(self, system, context):
        self.last_call = {}
        waited = self.pacer.wait()
        self.quota_wait_seconds += waited
        try:
            return super()._call(system, context)
        finally:
            self.last_call["quota_wait_seconds"] = waited


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="benchmarks/graph_pilot_questions.json")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    settings = Settings.from_env()
    if settings.llm_base_url.rstrip("/") != "https://api.groq.com/openai/v1":
        raise ValueError("This pacing wrapper requires the Groq endpoint")
    output = Path(args.output)
    pacer = TokenPacer(output/"quota_headers.jsonl")
    questions = json.loads(Path(args.questions).read_text())
    run_pilot(args.corpus, questions, settings, output,
              planner_factory=lambda: PacedPlanner(settings, pacer))
    experiment = json.loads((output/"experiment.json").read_text())
    experiment["pacing"] = "Shared token-reset pacing across both arms; no automatic retries; wait time measured separately"
    atomic_json(output/"experiment.json", experiment)
    analyze_pilot(output)


if __name__ == "__main__":
    main()
