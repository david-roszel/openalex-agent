"""Synthetic engineering checks for graph capacity; never retrieval evidence.

Run: python -m openalex_agent.graph_benchmark --papers 10000 --references 20
Every bibliographic string/ID in this fixture is synthetic. Only runtime,
memory and correctness measurements come from actual execution.
"""

import argparse
import gc
import json
from pathlib import Path
import platform
import resource
from time import perf_counter
from .graph import ScholarlyGraph
from .models import Paper, utcnow


def synthetic_paper(number: int, count: int, references: int) -> Paper:
    ids = [f"W{count + 1}"] + [f"W{count + 2 + ((number + j) % (count - 1))}" for j in range(references - 1)]
    raw = {"id": f"W{number}", "title": f"SYNTHETIC SCALE TEST paper {number}",
           "referenced_works": ids,
           "abstract_inverted_index": {"Synthetic": [0], "engineering": [1], "fixture": [2]},
           "authorships": [{"author": {"id": author}, "institutions": [{"id": "I1"}]}
                           for author in ("A1", f"A{number + 1}")],
           "topics": [{"id": f"T{number % 8 + 1}"}]}
    paper = Paper.from_openalex(raw, provenance="SYNTHETIC GRAPH CAPACITY FIXTURE ONLY")
    paper.retrieved_at = "2026-01-01T00:00:00+00:00"
    return paper


def run_profile(directory: str | Path, *, papers: int = 10000, references: int = 20, updates: int = 1000) -> dict:
    if type(papers) is not int or papers < 8 or not 2 <= references <= papers or not 0 <= updates <= papers:
        raise ValueError("Require at least 8 papers, 2..papers references, and 0..papers updates")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "synthetic-graph.json"
    timings = {}
    graph = ScholarlyGraph()
    start = perf_counter()
    for number in range(1, papers + 1):
        graph.add(synthetic_paper(number, papers, references), action="synthetic_import")
    timings["build_seconds"] = perf_counter() - start
    baseline = graph.stats()
    assert baseline["resolved_papers"] == papers
    assert baseline["paper_nodes"] == 2 * papers
    assert baseline["edges"] == papers * references + 4 * papers + 1
    start = perf_counter()
    for number in range(1, updates + 1):
        partial = Paper.from_openalex({"id": f"W{number}", "title": f"SYNTHETIC revised title {number}"},
                                      provenance="SYNTHETIC GRAPH CAPACITY FIXTURE ONLY")
        partial.retrieved_at = "2026-02-01T00:00:00+00:00"
        graph.add(partial, action="synthetic_partial_update")
        work = graph.paper(partial.id)
        assert len(work.references) == references and work.abstract and len(work.authorships) == 2
        assert len(graph.paper_history(partial.id)) == 1
    timings["update_seconds"] = perf_counter() - start
    assert graph.stats() == baseline
    # Canonical aliases must not create another node or downgrade newer data.
    alias = synthetic_paper(1, papers, references)
    alias.id = "http://openalex.org/w1"
    graph.add(alias, action="synthetic_import")
    assert graph.stats() == baseline
    assert graph.paper_state(f"W{papers + 1}") == "unresolved"
    assert graph.paper(f"W{papers + 1}") is None
    promotions = min(10, papers)
    for number in range(papers + 1, papers + promotions + 1):
        resolved = Paper.from_openalex({"id": f"W{number}", "title": "SYNTHETIC resolved target", "referenced_works": []},
                                       provenance="SYNTHETIC GRAPH CAPACITY FIXTURE ONLY")
        graph.add(resolved, action="synthetic_resolution")
    assert graph.stats()["paper_nodes"] == baseline["paper_nodes"]
    assert graph.stats()["resolved_papers"] == papers + promotions
    start = perf_counter()
    view = graph.neighborhood("W1", hops=2, max_nodes=25, relations=["PAPER_CITES_PAPER"])
    assert len(view["nodes"]) <= 25
    timings["neighborhood_seconds"] = perf_counter() - start
    expected = graph.stats()
    start = perf_counter()
    graph.save(path)
    timings["save_seconds"] = perf_counter() - start
    del graph
    gc.collect()
    start = perf_counter()
    restored = ScholarlyGraph.load(path)
    timings["load_seconds"] = perf_counter() - start
    assert restored.stats() == expected
    assert restored.paper_state(f"W{papers + 1}") == "retrieved"
    assert len(restored.paper_history("W1")) == (1 if updates else 0)
    assert restored.detailed_stats()["unverified_citations"] == 0
    assert restored.citation_graph(include_unresolved=False).number_of_nodes() == papers + promotions
    result = {"kind": "SYNTHETIC ENGINEERING VALIDATION; NOT RESEARCH RESULTS", "verified_at": utcnow(),
              "python": platform.python_version(), "papers_ingested": papers, "references_per_paper": references,
              "partial_updates": updates, "targets_promoted": promotions, "graph": restored.detailed_stats(),
              "checks_passed": True, "timings": timings, "snapshot_bytes": path.stat().st_size,
              "peak_process_rss_mib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    (directory / "engineering-validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", type=int, default=10000)
    parser.add_argument("--references", type=int, default=20)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--output", default="data/validation/graph-scale")
    args = parser.parse_args()
    print(json.dumps(run_profile(args.output, papers=args.papers, references=args.references, updates=args.updates)))


if __name__ == "__main__":
    main()
