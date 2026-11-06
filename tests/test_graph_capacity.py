"""Exercise 10,000 actual graph records, not just a large search-service index."""

from openalex_agent.graph_benchmark import run_profile


def test_ten_thousand_papers_updates_promotion_and_snapshot_roundtrip(tmp_path):
    result = run_profile(tmp_path, papers=10000, references=4, updates=100)
    assert result["checks_passed"]
    assert result["graph"]["resolved_papers"] == 10010
    assert result["graph"]["paper_nodes"] == 20000
    assert result["graph"]["edges"] == 80001
    assert result["graph"]["unresolved_papers"] == 9990
    assert result["graph"]["unverified_citations"] == 0
