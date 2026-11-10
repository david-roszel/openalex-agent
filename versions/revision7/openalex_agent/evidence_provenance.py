"""Claim-time access reconstruction; final corpus/state is never access proof."""

import json
from pathlib import Path

from .models import canonical_id


def signature(record):
    return tuple(record.get(k) for k in ("statement", "work_id", "evidence_quote", "insufficient"))


def claim_access(directory, claims):
    path = Path(directory) / "events.jsonl"
    timeline, observed = {}, {}
    complete = False
    if path.exists():
        expected_seq = 1
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                complete = False
                break
            if event.get("event_seq") != expected_seq:
                complete = False
                break
            expected_seq += 1
            if event.get("kind") == "start":
                complete = True
            if event.get("kind") == "observation":
                observed.update({p["id"]: p for p in event.get("papers", [])})
            if event.get("kind") == "evidence_record":
                identifier = event.get("work_id")
                timeline.setdefault(signature(event), []).append(dict(
                    source_observed_at_claim=identifier in observed if complete else None,
                    source_accessible_at_claim=None, source_record=observed.get(identifier),
                    basis="ordered execution events" if complete else "incomplete execution events"))
    result = []
    for claim in claims:
        try:
            identifier = canonical_id(claim.get("work_id"))
        except (ValueError, TypeError):
            identifier = None
        access = dict(source_observed_at_claim=None, source_accessible_at_claim=None,
                      source_record=None, basis="unknown; no claim-time provenance")
        matches = timeline.get(signature(claim), [])
        if matches:
            access.update(matches.pop(0))
        provenance = claim.get("provenance", {})
        if provenance.get("schema_version") == 1 and type(provenance.get("model_call")) is int:
            for key in ("source_observed_at_claim", "source_accessible_at_claim"):
                if type(provenance.get(key)) is bool:
                    access[key] = provenance[key]
            access.update(source_record=provenance.get("source_record"), basis="harness claim-time provenance")
        if identifier is None:
            access.update(source_observed_at_claim=False, source_accessible_at_claim=False)
        # A positive observation must have an actual source record, never just
        # an identifier-validity flag or the final state of the run.
        if access["source_observed_at_claim"] is True and not (
                isinstance(access["source_record"], dict) and access["source_record"].get("id") == identifier):
            access["source_observed_at_claim"] = None
        result.append(access)
    return result
