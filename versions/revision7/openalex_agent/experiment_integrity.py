"""Validate experiment identity before assessment or aggregation."""

from collections import Counter


CELL_KEYS = ("query_id", "action_budget", "trial", "method", "variant")


def validate_schedule(manifest, rows):
    controlled = manifest.get("suite") in ("controlled-five", "controlled-seven")
    if controlled and not {"status", "planned_runs", "run_order"} <= manifest.keys():
        raise ValueError("Controlled assessment requires a complete experiment schedule")
    if "planned_runs" in manifest and (manifest.get("status") != "complete" or len(rows) != manifest["planned_runs"]):
        raise ValueError("Experiment is incomplete; finish the scheduled runs before assessment")
    if "run_order" in manifest:
        try:
            expected = [tuple(r[k] for k in CELL_KEYS) for r in manifest["run_order"]]
            actual = [tuple(r[k] for k in CELL_KEYS) for r in rows]
        except KeyError as exc:
            raise ValueError("Missing experimental cell identity") from exc
        if len(set(actual)) != len(actual) or len(set(expected)) != len(expected):
            raise ValueError("Duplicate experimental trial/cell identity")
        if Counter(actual) != Counter(expected):
            raise ValueError("Experiment schedule does not match the recorded runs")
