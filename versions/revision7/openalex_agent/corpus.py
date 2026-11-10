"""Frozen OpenAlex collections, checkpointed acquisition, and offline retrieval."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Protocol
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .models import Paper, canonical_id, utcnow
from .openalex import OpenAlexClient, Usage


class Retriever(Protocol):
    usage: Usage
    fingerprint: str
    def search(self, query: str, limit: int = 20) -> list[Paper]: ...
    def get(self, paper_id: str) -> Paper: ...
    def references(self, paper_id: str, limit: int = 100) -> list[Paper]: ...
    def citations(self, paper_id: str, limit: int = 100) -> list[Paper]: ...
    def topic(self, topic_id: str, limit: int = 100) -> list[Paper]: ...


def atomic_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def freeze(papers: list[Paper], directory: str | Path, *, acquisition: list[dict]) -> dict:
    """Create an immutable snapshot, refusing to overwrite an existing manifest."""
    directory = Path(directory)
    if (directory / "manifest.json").exists():
        raise ValueError("Corpus already frozen; choose a new output directory")
    directory.mkdir(parents=True, exist_ok=True)
    papers = sorted({p.id: p for p in papers}.values(), key=lambda p: p.id)
    data = "".join(json.dumps(p.to_dict(), sort_keys=True, ensure_ascii=False) + "\n" for p in papers).encode()
    path = directory / "papers.jsonl"
    temporary = directory / "papers.jsonl.tmp"
    temporary.write_bytes(data)
    temporary.replace(path)
    manifest = {"schema": 1, "created_at": utcnow(), "count": len(papers),
                "sha256": hashlib.sha256(data).hexdigest(), "ids": [p.id for p in papers],
                "acquisition": acquisition}
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def acquire(client: OpenAlexClient, queries: list[str], directory: str, *, per_query: int = 1000,
            filters: str = "") -> dict:
    """Resume at the first incomplete page; completed cached pages never redownload."""
    if per_query < 1 or not queries or any(not q.strip() for q in queries):
        raise ValueError("Acquisition needs nonempty queries and a positive per-query limit")
    if (Path(directory) / "manifest.json").exists():
        return LocalCorpus(directory).manifest
    selected: dict[str, Paper] = {}
    records = []
    for query in queries:
        job_key = "acquire:" + hashlib.sha256(json.dumps([query, filters, 100]).encode()).hexdigest()
        state = client.db.job_get(job_key) or {"ids": [], "cursor": "*", "started_at": utcnow(), "pages": 0}
        while state["cursor"] and len(state["ids"]) < per_query:
            papers, next_cursor = client.page(query=query, filters=filters, cursor=state["cursor"], per_page=100)
            state["ids"] = list(dict.fromkeys(state["ids"] + [p.id for p in papers]))
            if next_cursor == state["cursor"]:
                raise ValueError("OpenAlex returned a repeated cursor")
            state["cursor"] = next_cursor if papers else None
            state["pages"] += 1
            state["updated_at"] = utcnow()
            client.db.job_put(job_key, state)
        for paper_id in state["ids"][:per_query]:
            paper = client.db.get_paper(paper_id)
            if paper is None:
                raise ValueError("Acquisition checkpoint has missing metadata; restore database")
            selected[paper.id] = paper
        records.append({"query": query, "filters": filters, "per_query": per_query, **state})
    manifest = freeze(list(selected.values()), directory, acquisition=records)
    client.db.event(client.run_id, "corpus_frozen", {"sha256": manifest["sha256"], "count": manifest["count"],
                                                   "usage": asdict(client.usage)})
    return manifest


class LocalCorpus:
    """Same metadata/action surface for every method; no relevance judgments.

    The full collection is available only to this search service, never to agent
    context or graph memory. The index and citation lookup are shared equally.
    """
    def __init__(self, directory: str | Path):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        data = (directory / "papers.jsonl").read_bytes()
        self.fingerprint = hashlib.sha256(data).hexdigest()
        if self.fingerprint != self.manifest["sha256"]:
            raise ValueError("Corpus checksum mismatch")
        papers = [Paper.from_dict(json.loads(line)) for line in data.splitlines() if line.strip()]
        self.by_id = {p.id: p for p in papers}
        if len(papers) != len(self.by_id) or sorted(self.by_id) != sorted(self.manifest["ids"]) or len(papers) != self.manifest["count"]:
            raise ValueError("Corpus IDs/count do not match manifest")
        self.papers = sorted(papers, key=lambda p: p.id)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=200000)
        try:
            self.matrix = self.vectorizer.fit_transform([p.text for p in self.papers])
        except ValueError:
            self.matrix = None
        self.usage = Usage()
        self.citing: dict[str, list[Paper]] = {}
        self.topics: dict[str, list[Paper]] = {}
        for p in self.papers:
            for ref in p.references:
                self.citing.setdefault(ref, []).append(p)
            for topic in p.topics:
                self.topics.setdefault(topic.id, []).append(p)

    def search(self, query: str, limit: int = 20) -> list[Paper]:
        if limit < 1 or not query.strip():
            raise ValueError("Provide a nonempty query and positive limit")
        if self.matrix is None:
            return []
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        return [self.papers[i] for i in np.argsort(-scores, kind="stable")[:limit] if scores[i] > 0]

    def get(self, paper_id: str) -> Paper:
        try:
            return self.by_id[canonical_id(paper_id)]
        except KeyError:
            raise ValueError("Paper is outside the fixed corpus") from None

    def references(self, paper_id: str, limit: int = 100) -> list[Paper]:
        return [self.by_id[i] for i in self.get(paper_id).references if i in self.by_id][:limit]

    def related(self, paper_id: str, limit: int = 100) -> list[Paper]:
        return [self.by_id[i] for i in self.get(paper_id).related if i in self.by_id][:limit]

    def citations(self, paper_id: str, limit: int = 100) -> list[Paper]:
        return self.citing.get(canonical_id(paper_id), [])[:limit]

    def topic(self, topic_id: str, limit: int = 100) -> list[Paper]:
        return self.topics.get(canonical_id(topic_id, "T"), [])[:limit]
