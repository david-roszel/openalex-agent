"""Synthetic scoring capacity check. Produces no retrieval-quality results."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time

from .corpus import atomic_json
from .experiments import environment_record
from .graph import ScholarlyGraph
from .graph_benchmark import synthetic_paper
from .selection import NOVELTY_BATCH_SIZE, SELECTION_VERSION, Scoring, score_candidates


def run_profile(count: int, output: Path) -> dict:
    if type(count) is not int or count < 8:
        raise ValueError("Use at least eight synthetic papers")
    graph = ScholarlyGraph()
    papers = {}
    for number in range(1, count + 1):
        paper = synthetic_paper(number, count, 2)
        papers[paper.id] = paper
        graph.add(paper, action="synthetic_scoring_capacity")
    scoring = Scoring()
    start = time.perf_counter()
    rows = score_candidates("synthetic engineering fixture", papers, graph, scoring)
    seconds = time.perf_counter() - start
    assert len(rows) == 2 * count and sum(r["resolved"] for r in rows) == count
    assert all(0 <= value <= 1 for row in rows for value in row["components"].values())
    assert rows == sorted(rows, key=lambda row: (-row["score"], row["id"]))
    record = {"kind": "SYNTHETIC ENGINEERING CHECK; NOT RESEARCH RESULTS", "environment": environment_record(),
              "selection_version": SELECTION_VERSION, "scoring": asdict(scoring), "observed_papers": count,
              "unresolved_targets": count, "scored_candidates": len(rows), "graph": graph.stats(),
              "novelty_batch_rows": NOVELTY_BATCH_SIZE, "score_seconds": seconds,
              "peak_process_rss_mib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "checks_passed": True, "importance_method": rows[0]["importance_method"]}
    atomic_json(output, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("data/milestone4-validation/selection-scale.json"))
    args = parser.parse_args()
    print(json.dumps(run_profile(args.papers, args.output)))


if __name__ == "__main__":
    main()
