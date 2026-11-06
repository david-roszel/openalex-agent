"""Architectural scale smoke test only. No synthetic research results are saved."""

from openalex_agent.controller import Budget, Controller
from openalex_agent.corpus import LocalCorpus, freeze
from openalex_agent.models import Paper


def test_ten_thousand_candidate_index_keeps_agent_observations_bounded(tmp_path, db):
    papers = [Paper(id=f"https://openalex.org/W{i+1}", title=f"SYNTHETIC TEST retrieval document {i}",
                    provenance="SYNTHETIC TEST ONLY") for i in range(10000)]
    directory = tmp_path / "corpus"
    freeze(papers, directory, acquisition=[{"source": "SYNTHETIC SCALE TEST ONLY"}])
    corpus = LocalCorpus(directory)
    controller = Controller("retrieval", corpus, db, output=str(tmp_path / "runs"),
                            budget=Budget(max_actions=2, max_inspected=5, page_size=3, max_context_chars=4000))
    result = controller.run()
    assert len(corpus.by_id) == 10000
    assert len(result["inspected_ids"]) <= 5
    assert controller.graph.stats()["resolved_papers"] <= 5
    assert len(__import__("json").dumps(controller.context(), ensure_ascii=False)) <= 4000
