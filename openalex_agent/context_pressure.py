"""Measured token-pressure engineering diagnostics; no hosted inference.

The fixed script exercises each available mechanism, not learned effectiveness.
Both policies receive the same stream and late recovery task. No result requires
CLM to win. Conventional top-eight/recent-four selection is left intact.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import re
from types import SimpleNamespace

from .actions import make_action
from .context_memory import serialized
from .context_planner import ContextPlanner, EngineeringContextModel, InferenceLimits
from .controlled import ControlledController, GRAPH_FIELDS
from .controller import Budget
from .corpus import LocalCorpus, atomic_json, freeze
from .models import Paper, utcnow
from .storage import Database
from .token_budget import count, metadata, input_tokens
from .research_metrics import context_metrics

CONDITIONS = {"low": 4, "medium": 32, "high": 256}
NEEDLE = re.compile(r"FACT\[W\d+:datum-\d+\]")


def stimulus(blocks, seed, *, evidence_position=0, distractor_chars=350, correction_at=None):
    rng = random.Random(seed)
    papers = []
    for i in range(blocks):
        # Descending, equal-length canonical IDs make each new tied-relevance
        # record enter the unchanged lexical/id selection policy. This ensures
        # measured input delivery, rather than counting unseen corpus papers.
        identifier = f"W{199999-i}"
        p = Paper.from_openalex({"id": identifier, "title": "retrieval"}, provenance="SYNTHETIC TOKEN PRESSURE ONLY")
        fact = f"FACT[{identifier}:datum-{seed+73}] " if i == evidence_position else ""
        p.abstract = (fact + " ".join(rng.choice("qzxvkwj") for _ in range(200)))[:distractor_chars]
        if i==correction_at:
            original=papers[evidence_position]
            p=Paper.from_dict(original.to_dict())
            p.retrieved_at=utcnow()
            p.metadata_fields=list(dict.fromkeys((p.metadata_fields or [])+["abstract"]))
            p.abstract=(f"FACT[{original.id.rsplit('/',1)[-1]}:datum-{seed+74}] "+"CORRECTION replaces the earlier datum. "+p.abstract)[:distractor_chars]
        papers.append(p)
    return papers


class PressureStream(LocalCorpus):
    def __init__(self, corpus, papers):
        super().__init__(corpus)
        self.stream, self.index = papers, 0

    def search(self, query, limit=20):
        result = self.stream[self.index:self.index+1]
        self.index += 1
        return result


class PressurePolicy(EngineeringContextModel):
    """One declared script, common probe and recovery rules in both modes."""
    def __init__(self, horizon, active_limit, *, archive_reads=True):
        super().__init__()
        self.horizon, self.active_limit = horizon, active_limit
        self.read_start = 0
        self.archive_reads=archive_reads
        self.settings = SimpleNamespace(llm_max_tokens=512)
        self.configuration = {**self.configuration, "pressure_policy": "overlapping-archive-read-v2"}

    def respond(self,system,payload,output):
        before_in=self.usage.estimated_input_tokens;before_out=self.usage.estimated_output_tokens
        value=super().respond(system,payload,output)
        # The pressure harness has a pinned tokenizer: use its exact counts for
        # cumulative preflight as well as plots, rather than byte/4 proxies.
        self.usage.estimated_input_tokens=before_in+input_tokens(system,payload)
        self.usage.estimated_output_tokens=before_out+count(serialized(output))
        return value

    def _call(self, system, payload):
        active = payload.get("active_context", serialized(payload.get("observation", {})))
        if payload["protected"]["context_management"] == "clm" and (
                payload.get("pending_observation_chars") or count(active) > self.active_limit*.75):
            value = {"context_edit": {"path": "active_context.txt", "op": "replace",
                     "text": "\n".join(dict.fromkeys(NEEDLE.findall(active)))}}
        elif payload["protected"]["round"] >= self.horizon:
            visible = active + serialized(payload.get("recovered_observation", {}))
            fact = NEEDLE.search(visible)
            if not fact and self.archive_reads and self.read_start < 4096:
                value = {"read_observation": {"index": 0, "start": self.read_start, "chars": 512}}
                # A probe can straddle a chunk boundary. Overlap applies in
                # both modes and does not retain hidden evidence between calls.
                self.read_start += 256
            else:
                quote = fact.group(0) if fact else ""
                value = {"decision": {"action": "STOP_RETRIEVAL", "arguments": {}}, "evidence_records": [{
                    "statement": quote or "Insufficient available evidence", "work_id": quote[5:].split(":")[0] if fact else None,
                    "evidence_quote": quote, "insufficient": not bool(fact)}]}
        else:
            value = {"decision": make_action("SEARCH_PAPERS", f"retrieval stream {payload['protected']['round']}").to_dict()}
        return self.respond(system, payload, value)


def run_pressure(condition, directory, *, seed=0, active_tokens=4096, observations=None,
                 evidence_position=0, distractor_chars=350, correction_at=None, archive_reads=True,
                 architectures=("llm","clm"), return_invalid=False, cancel_check=None, global_token_limit=None,
                 full_input_tokens=None, per_arm_tokens=10000000):
    if condition not in CONDITIONS:
        raise ValueError("Choose low, medium or high pressure")
    directory = Path(directory)
    if directory.exists():
        raise ValueError("Use a fresh pressure diagnostic directory")
    blocks, ceiling = observations if observations is not None else CONDITIONS[condition], active_tokens
    if (type(blocks) is not int or not 2<=blocks<=1000 or type(ceiling) is not int or not 512<=ceiling<=16000
        or type(evidence_position) is not int or not 0<=evidence_position<blocks or not 80<=distractor_chars<=350
        or correction_at is not None and not evidence_position<correction_at<blocks
        or not architectures or len(set(architectures))!=len(architectures) or any(a not in ("llm","llm_graph","clm","clm_graph") for a in architectures)):
        raise ValueError("Invalid pressure diagnostic configuration")
    papers = stimulus(blocks, seed,evidence_position=evidence_position,distractor_chars=distractor_chars,correction_at=correction_at)
    needle = NEEDLE.search(papers[evidence_position].abstract).group(0)
    expected_needle=NEEDLE.search(papers[correction_at].abstract).group(0) if correction_at is not None else needle
    corpus = directory / "corpus"
    freeze(papers, corpus, acquisition=[{"source": "SYNTHETIC DIAGNOSTIC; NO QUALITY CLAIM", "seed": seed}])
    limits = InferenceLimits(max_total_tokens=per_arm_tokens, max_active_tokens=ceiling,
        max_input_tokens=full_input_tokens if full_input_tokens is not None else max(8192,ceiling+4096))
    rows, archive_signatures = [], []
    used_tokens=0
    for architecture in architectures:
        if cancel_check and cancel_check():
            from .execution_control import ExperimentStopped
            raise ExperimentStopped()
        mode="clm" if architecture.startswith("clm") else "conventional"
        remaining=None if global_token_limit is None else global_token_limit-used_tokens
        if remaining is not None and remaining<=0:break
        arm_limits=limits if remaining is None else InferenceLimits(max_total_tokens=min(limits.max_total_tokens,remaining),
            max_active_tokens=limits.max_active_tokens,max_input_tokens=limits.max_input_tokens)
        model = PressurePolicy(blocks, ceiling,archive_reads=archive_reads)
        planner = ContextPlanner(model, mode=mode, limits=arm_limits,archive_reads=archive_reads)
        events = []
        with Database(":memory:") as db:
            controller = ControlledController("retrieval", PressureStream(corpus, papers), db,
                architecture=architecture, planner=planner,
                budget=Budget(max_actions=blocks+1, max_rounds=blocks+1, max_inspected=blocks+1,
                    max_llm_calls=blocks*4+16, patience=blocks+1, page_size=1, max_context_chars=64000),
                output=str(directory / (mode if architectures==("llm","clm") else architecture)))
            result = controller.run()
            events = db.events(controller.run_id)
        requests = [json.loads(p.read_text()) for p in sorted(planner.store.directory.glob("request-*.json"))]
        used_tokens+=sum(r["study_tokens"]["full_input_tokens"]+count(serialized(r.get("output"))) for r in requests)
        observations = planner.store.observations
        common=[{k:v for k,v in o.items() if k not in GRAPH_FIELDS} for o in observations]
        archive_signatures.append(hashlib.sha256(serialized(common).encode()).hexdigest())
        # Count incremental selected input, not repeated whole top-eight views.
        delta_policy = SimpleNamespace(_seen={})
        observation_tokens = sum(count(serialized(ContextPlanner._delta(delta_policy, body))) for body in observations)
        source_ids = {p["id"] for body in observations for p in body.get("top_papers", [])}
        active = [r["input"].get("active_context", serialized(r["input"].get("observation", {}))) for r in requests]
        final_active=active[-1] if active else ""
        recovered = [r["input"]["recovered_observation"]["text"] for r in requests if "recovered_observation" in r["input"]]
        management = [r for r in requests if "context_edit" in (r.get("output") or {}) or "read_observation" in (r.get("output") or {})]
        recovery_attempts=[e for e in events if e["kind"]=="context_recovery"]
        recovered_probe=any(needle in text for text in recovered)
        versions={(p.id,p.title,p.abstract[:350]) for p in papers}
        delivered={(p["id"],p["title"],p["abstract_excerpt"]) for o in observations for p in o.get("top_papers",[])}
        rows.append(dict(mode=mode, architecture=architecture,directory=result["directory"], model_configuration=model.configuration,
            limits=asdict(arm_limits), response_reserve_tokens=512, observations=len(observations),
            total_observation_tokens_encountered=observation_tokens,
            unique_delivered_source_excerpt_tokens=sum(count(title+"\n"+abstract) for identifier,title,abstract in versions if (identifier,title,abstract) in delivered),
            active_context_capacity_tokens=ceiling, active_context_tokens_used=[r["study_tokens"]["active_tokens"] for r in requests],
            max_active_context_tokens=max((r["study_tokens"]["active_tokens"]+r["study_tokens"]["recovery_tokens"] for r in requests),default=0),
            max_full_input_tokens=max((r["study_tokens"]["full_input_tokens"] for r in requests),default=0),
            all_stream_sources_delivered=versions<=delivered,
            evidence_retained_in_final_active=needle in final_active,
            evidence_omitted_from_selected_view=any(needle not in serialized(o) for o in observations),
            evidence_removed_from_active=any(needle in a and needle not in b for a,b in zip(active,active[1:])),
            evidence_recoverable_externally=archive_reads and any(needle in serialized(o) for o in observations),
            evidence_exists_in_audit_archive=any(needle in serialized(o) for o in observations),
            evidence_successfully_recovered=recovered_probe,
            exact_recovery_accuracy=context_metrics([needle],"",recovered={needle:needle if recovered_probe else ""})["exact_recovery_accuracy"] if recovery_attempts else None,
            recovery_target="initial streamed probe; correction correctness is scored separately",
            final_supported_synthetic_answer=any(r["statement"]==expected_needle and not r["insufficient"] for r in result["evidence_records"]),
            factual_errors=sum(not r["insufficient"] and r["statement"]!=expected_needle for r in result["evidence_records"]),
            current_evidence_retained=expected_needle in final_active,
            model_calls=len(requests), management_calls=len(management),
            management_study_input_tokens=sum(r["study_tokens"]["full_input_tokens"] for r in management),
            management_study_output_tokens=sum(count(serialized(r["output"])) for r in management),
            management_seconds=sum(e.get("seconds",0) for e in events if e["kind"]=="context_model_call" and e.get("operation_kind") in ("edit","recovery")),
            context_adapter_local_seconds=result["context_adapter_local_seconds"],
            edit_operations=[r for r in planner.store.edits if r["source"]=="model"],
            recovery_operations=[e for e in events if e["kind"]=="context_recovery"],
            actual_provider_tokens=None, actual_provider_cost_usd=None,
            stop_reason=result["stop_reason"], requests_all_within_limits=all(planner.payload_fits(r["input"],system=r["system"]) for r in requests)))
    ratio = min((r["total_observation_tokens_encountered"] for r in rows),default=0)/ceiling
    source_ratio = min((r["unique_delivered_source_excerpt_tokens"] for r in rows),default=0)/ceiling
    valid = all(r["all_stream_sources_delivered"] and r["requests_all_within_limits"] and r["observations"]==blocks for r in rows)
    valid = valid and ((ratio<1) if condition=="low" else (ratio>1) if condition=="medium" else (ratio>8))
    valid = valid and ((source_ratio<1) if condition=="low" else (source_ratio>1) if condition=="medium" else (source_ratio>8))
    valid = valid and len(set(archive_signatures))==1 and len(rows)==len(architectures)
    valid = valid and all(r["stop_reason"]=="planner_stop" for r in rows)
    from .git_provenance import git_provenance
    report = dict(kind="MEASURED OFFLINE TOKEN DIAGNOSTIC; NOT MODEL PERFORMANCE", condition=condition, seed=seed, git=git_provenance(),
        tokenizer=metadata(), shared_observation_archives= len(set(archive_signatures))==1,
        pressure_ratio=ratio, unique_source_pressure_ratio=source_ratio, pressure_valid=valid, rows=rows,
        total_study_tokens_used=used_tokens,global_token_limit=global_token_limit,
        configuration=dict(observations=blocks,active_tokens=ceiling,evidence_position=evidence_position,
            distractor_chars=distractor_chars,correction_at=correction_at,archive_reads=archive_reads,architectures=architectures),
        limitation="Fixed scripted choices, synthetic evidence; no claim of learned or real-paper effectiveness")
    atomic_json(directory / "pressure.json", report)
    if not valid and not return_invalid:
        raise ValueError(f"Pressure condition invalid; inspect {directory / 'pressure.json'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = run_pressure(args.condition, args.output, seed=args.seed)
    print(serialized({"condition":args.condition,"pressure_valid":report["pressure_valid"],"pressure_ratio":report["pressure_ratio"]}))


if __name__ == "__main__":
    main()
