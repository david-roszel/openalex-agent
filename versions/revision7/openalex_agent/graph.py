"""Incremental scholarly graph, evidence maintenance, persistence and traversal."""

from collections import Counter, defaultdict, deque
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable
import networkx as nx
from .models import Paper, canonical_id, utcnow, merge_observation, MERGE_FIELDS, OPENALEX_SOURCES


RELATIONS = {
    "PAPER_CITES_PAPER": ("paper", "paper"),
    "PAPER_WRITTEN_BY_AUTHOR": ("paper", "author"),
    "PAPER_HAS_TOPIC": ("paper", "topic"),
    "AUTHOR_AFFILIATED_WITH_INSTITUTION": ("author", "institution"),
}
NODE_KINDS = {"paper": "W", "author": "A", "topic": "T", "institution": "I"}
RELATION_FIELDS = {"PAPER_CITES_PAPER": "references", "PAPER_WRITTEN_BY_AUTHOR": "authorships",
                   "AUTHOR_AFFILIATED_WITH_INSTITUTION": "authorships", "PAPER_HAS_TOPIC": "topics"}
SOURCE_FIELDS = {"references": "referenced_works", "authorships": "authorships", "topics": "topics"}


def entity_id(value: str) -> str:
    """Normalize a supported scholarly ID without guessing from names."""
    for kind in NODE_KINDS.values():
        try:
            return canonical_id(value, kind)
        except ValueError:
            pass
    raise ValueError("Expected an OpenAlex paper, author, topic or institution ID")


def _timestamp(value: str) -> str:
    try:
        date = datetime.fromisoformat(value)
        if date.tzinfo is None:
            raise ValueError("Missing timezone")
        return date.astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError):
        raise ValueError("Graph metadata requires an ISO timestamp with a timezone") from None


def _paper_copy(paper: Paper) -> Paper:
    """Canonicalize a copy before mutation, including direct Python API inputs."""
    try:
        copy = Paper.from_dict(paper.to_dict())
        copy.id = canonical_id(copy.id)
        copy.references = list(dict.fromkeys(canonical_id(i) for i in copy.references))
        copy.related = list(dict.fromkeys(canonical_id(i) for i in copy.related))
        copy.retrieved_at = _timestamp(copy.retrieved_at)
        for authorship in copy.authorships:
            if authorship.author:
                authorship.author.id = canonical_id(authorship.author.id, "A")
            for institution in authorship.institutions:
                institution.id = canonical_id(institution.id, "I")
        for topic in copy.topics:
            topic.id = canonical_id(topic.id, "T")
        entities = [t for t in copy.topics]
        entities += [a.author for a in copy.authorships if a.author]
        entities += [i for a in copy.authorships for i in a.institutions]
        if any(e.name is not None and not isinstance(e.name, str) for e in entities):
            raise ValueError("Entity names must be text or unavailable")
        if not isinstance(copy.provenance, str) or not copy.provenance:
            raise ValueError("Missing source provenance")
        if copy.metadata_source is not None and not isinstance(copy.metadata_source, str):
            raise ValueError("Invalid metadata source")
        if copy.metadata_fields is not None and (not isinstance(copy.metadata_fields, list) or
                any(not isinstance(f, str) or f not in MERGE_FIELDS for f in copy.metadata_fields)):
            raise ValueError("Invalid observed metadata fields")
        return copy
    except (AttributeError, KeyError, TypeError):
        raise ValueError("Malformed paper metadata for graph construction") from None


def _relationships(paper: Paper) -> set[tuple[str, str, str]]:
    result = {(paper.id, ref, "PAPER_CITES_PAPER") for ref in paper.references}
    for authorship in paper.authorships:
        if authorship.author:
            author = authorship.author.id
            result.add((paper.id, author, "PAPER_WRITTEN_BY_AUTHOR"))
            result.update((author, i.id, "AUTHOR_AFFILIATED_WITH_INSTITUTION") for i in authorship.institutions)
    result.update((paper.id, t.id, "PAPER_HAS_TOPIC") for t in paper.topics)
    return result


def _node_link(graph: nx.MultiDiGraph) -> dict:
    """Use an explicit portable edge field instead of NetworkX version defaults."""
    return deepcopy({"directed": True, "multigraph": True, "graph": dict(graph.graph),
        "nodes": [{**attrs, "id": node} for node, attrs in sorted(graph.nodes(data=True))],
        "edges": [{**attrs, "evidence": sorted(attrs["evidence"]), "source": u, "target": v, "key": k}
                  for u, v, k, attrs in sorted(graph.edges(keys=True, data=True), key=lambda e: e[:3])]})


class ScholarlyGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph(scholarly_schema=2)
        self._indexed_graph = None
        self._owners: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
        self._children: dict[str, set[str]] = defaultdict(set)
        self._discovery_keys: dict[str, set[tuple]] = defaultdict(set)
        self._evidence_sets: dict[tuple[str, str, str], set[str]] = {}
        self._reference_sets: dict[str, set[str]] = {}

    def _index(self) -> None:
        # Pre-existing controller checkpoints can replace the NetworkX instance.
        if self._indexed_graph is self.graph:
            return
        self._owners = defaultdict(set)
        self._children = defaultdict(set)
        self._discovery_keys = defaultdict(set)
        self._evidence_sets = {}
        self._reference_sets = {}
        for u, v, key, attrs in self.graph.edges(keys=True, data=True):
            self._evidence_sets[u, v, key] = set(attrs.get("evidence", []))
            for paper_id in attrs.get("evidence", []):
                self._owners[paper_id].add((u, v, key))
        for node, attrs in self.graph.nodes(data=True):
            metadata = attrs.get("metadata") or {}
            if metadata:
                self._reference_sets[node] = set(metadata.get("references", []))
            self._discovery_keys[node] = {(d.get("action"), d.get("parent"), d.get("depth"))
                                          for d in attrs.get("discoveries", [])}
            if attrs.get("type") == "paper" and attrs.get("parent"):
                self._children[attrs["parent"]].add(node)
        self._indexed_graph = self.graph

    def _offer_path(self, node: str, parent: str | None, depth: int, action: str, now: str) -> None:
        if node not in self.graph:
            archived = self.graph.graph.get("retired_entities", {}).pop(node, None)
            if archived:
                old_parent = archived.get("parent")
                archived["depth"] = self.graph.nodes[old_parent]["depth"] + 1 if old_parent else 0
                self.graph.add_node(node, **archived)
                if old_parent:
                    self._children[old_parent].add(node)
                self._discovery_keys[node] = {(d["action"], d.get("parent"), d["depth"])
                                               for d in archived.get("discoveries", [])}
        if node not in self.graph:
            self.graph.add_node(node, type="paper", resolved=False, status="unresolved", discovered_at=now,
                                parent=parent, depth=depth, action=action, discoveries=[])
            if parent:
                self._children[parent].add(node)
        attrs = self.graph.nodes[node]
        discovery = {"action": action, "parent": parent, "depth": depth}
        key = (action, parent, depth)
        if key not in self._discovery_keys[node]:
            attrs.setdefault("discoveries", []).append({**discovery, "observed_at": now})
            self._discovery_keys[node].add(key)
        if depth >= attrs.get("depth", depth):
            return
        if attrs.get("parent"):
            self._children[attrs["parent"]].discard(node)
        if parent:
            self._children[parent].add(node)
        attrs.update(parent=parent, depth=depth, action=action)
        queue = deque([node])
        while queue:
            ancestor = queue.popleft()
            for child in sorted(self._children[ancestor]):
                child_attrs = self.graph.nodes[child]
                child_attrs["depth"] = self.graph.nodes[ancestor]["depth"] + 1
                queue.append(child)

    def _entity(self, entity, kind: str, paper: Paper, field: str = "authorships") -> None:
        if entity.id not in self.graph:
            archived = self.graph.graph.get("retired_entities", {}).pop(entity.id, None)
            if archived:
                self.graph.add_node(entity.id, **archived)
        old = self.graph.nodes.get(entity.id, {})
        observed_at = self.graph.nodes[paper.id]["field_sources"][field]["retrieved_at"]
        self.graph.add_node(entity.id, type=kind)
        if entity.name and observed_at >= old.get("name_observed_at", ""):
            self.graph.nodes[entity.id].update(name=entity.name, name_source=paper.id,
                                             name_observed_at=observed_at)

    def _edge(self, edge: tuple[str, str, str], paper: Paper, now: str) -> None:
        source, target, relation = edge
        attrs = self.graph.get_edge_data(source, target, relation)
        if attrs is None:
            self.graph.add_edge(source, target, key=relation, relation=relation, evidence=[], observations={},
                                provenance="scholarly metadata", first_observed_at=now)
            attrs = self.graph.edges[edge]
            self._evidence_sets[edge] = set()
        if paper.id not in self._evidence_sets[edge]:
            attrs["evidence"].append(paper.id)
            self._evidence_sets[edge].add(paper.id)
        attrs["observations"][paper.id] = dict(self.graph.nodes[paper.id]["field_sources"][RELATION_FIELDS[relation]])
        attrs.update(last_observed_at=now, source_field=SOURCE_FIELDS[RELATION_FIELDS[relation]])
        self._owners[paper.id].add(edge)

    @staticmethod
    def _merge(paper: Paper, previous: dict | None, sources: dict) -> tuple[dict, dict]:
        return merge_observation(paper, previous, sources)

    def add(self, paper: Paper, *, action: str = "import", parent: str | None = None, depth: int | None = None) -> bool:
        """Merge an observation without discarding prior metadata observations.

        Returns True on first resolution. Discovery paths are independent of
        citation direction and metadata freshness.
        """
        paper = _paper_copy(paper)
        if paper.references and paper.metadata_source not in OPENALEX_SOURCES:
            raise ValueError("Citations require OpenAlex referenced_works metadata, not inferred relationships")
        self._index()
        if not isinstance(action, str) or not action.strip():
            raise ValueError("Discovery action must be nonempty text")
        if parent:
            parent = canonical_id(parent)
            if parent == paper.id or parent not in self.graph or not self.graph.nodes[parent].get("resolved"):
                raise ValueError("Discovery parent must be a different resolved paper")
        expected_depth = self.graph.nodes[parent]["depth"] + 1 if parent else 0
        if depth is None:
            depth = expected_depth
        if type(depth) is not int or depth != expected_depth:
            raise ValueError("Discovery depth must match its parent path")
        node = self.graph.nodes.get(paper.id, {})
        is_new = not node.get("resolved", False)
        old_metadata = node.get("metadata")
        now = utcnow()
        self._offer_path(paper.id, parent, depth, action, now)
        if node.get("retrieved_at") and paper.retrieved_at < _timestamp(node["retrieved_at"]):
            return is_new
        digest = hashlib.sha256(json.dumps(paper.to_dict(), sort_keys=True).encode()).hexdigest()
        if old_metadata == paper.to_dict() or node.get("last_observation_digest") == digest:
            return is_new
        field_sources = deepcopy(node.get("field_sources", {}))
        merged, field_sources = self._merge(paper, old_metadata, field_sources)
        paper = Paper.from_dict(merged)
        wanted = _relationships(paper)
        touched = set()
        for edge in self._owners[paper.id] - wanted:
            attrs = self.graph.edges[edge]
            attrs["evidence"] = [p for p in attrs["evidence"] if p != paper.id]
            self._evidence_sets[edge].discard(paper.id)
            attrs.get("observations", {}).pop(paper.id, None)
            if not attrs["evidence"]:
                self.graph.remove_edge(*edge)
                self._evidence_sets.pop(edge, None)
                touched.update(edge[:2])
        self._owners[paper.id].intersection_update(wanted)
        attrs = self.graph.nodes[paper.id]
        if is_new:
            attrs.update(resolved_at=now, resolved_by=action)
        if old_metadata:
            attrs.setdefault("metadata_history", []).append({"metadata": old_metadata,
                "field_sources": attrs.get("field_sources", {}), "superseded_at": now})
        attrs.update(resolved=True, status="retrieved", retrieved_at=paper.retrieved_at, metadata=merged,
                     metadata_provenance=paper.provenance, field_sources=field_sources,
                     last_observation_digest=digest)
        self._reference_sets[paper.id] = set(paper.references)
        if paper.title is not None:
            attrs["title"] = paper.title
            attrs["title_observed_at"] = field_sources["title"]["retrieved_at"]
        for reference in paper.references:
            self._offer_path(reference, paper.id, attrs["depth"] + 1, "reference", now)
        for authorship in paper.authorships:
            if authorship.author:
                self._entity(authorship.author, "author", paper)
                for institution in authorship.institutions:
                    self._entity(institution, "institution", paper)
        for topic in paper.topics:
            self._entity(topic, "topic", paper, "topics")
        for edge in sorted(wanted):
            self._edge(edge, paper, now)
        for node_id in touched:
            if node_id in self.graph and not self.graph.degree(node_id) and not self.graph.nodes[node_id].get("resolved") and not self._children[node_id]:
                old_parent = self.graph.nodes[node_id].get("parent")
                if old_parent:
                    self._children[old_parent].discard(node_id)
                self.graph.graph.setdefault("retired_entities", {})[node_id] = deepcopy(self.graph.nodes[node_id])
                self.graph.remove_node(node_id)
                self._children.pop(node_id, None)
                self._discovery_keys.pop(node_id, None)
        return is_new

    def paper_state(self, paper_id: str) -> str | None:
        """Retrieval means an observed metadata record, not full-text availability."""
        attrs = self.graph.nodes.get(canonical_id(paper_id))
        if attrs is None:
            if canonical_id(paper_id) in self.graph.graph.get("retired_entities", {}):
                return "unresolved"
            return None
        if attrs.get("resolved") and not attrs.get("metadata"):
            return "legacy_metadata_missing"
        return "retrieved" if attrs.get("resolved") else "unresolved"

    def paper_history(self, paper_id: str) -> list[dict]:
        return deepcopy(self.graph.nodes.get(canonical_id(paper_id), {}).get("metadata_history", []))

    def historical_entity(self, node_id: str) -> dict | None:
        """Inspect retained entity details after current evidence is superseded."""
        return deepcopy(self.graph.graph.get("retired_entities", {}).get(entity_id(node_id)))

    def _citation_verified(self, source: str, target: str) -> bool:
        """Legacy structural edges stay stored but cannot masquerade as evidence."""
        attrs = self.graph.nodes[source]
        metadata = attrs.get("metadata") or {}
        source_record = attrs.get("field_sources", {}).get("references", {})
        origin = source_record.get("metadata_source", metadata.get("metadata_source"))
        if origin is None and "metadata_source" not in metadata and metadata.get("provenance") == "OpenAlex API":
            origin = "openalex_legacy"
        return origin in OPENALEX_SOURCES and target in self._reference_sets.get(source, set())

    def paper(self, paper_id: str) -> Paper | None:
        """Return a copy of the observation, without network or database access."""
        metadata = self.graph.nodes.get(canonical_id(paper_id), {}).get("metadata")
        return Paper.from_dict(deepcopy(metadata)) if metadata else None

    def citation_graph(self, *, include_unresolved: bool = True) -> nx.DiGraph:
        self._index()
        graph = nx.DiGraph()
        graph.add_nodes_from(n for n, a in self.graph.nodes(data=True) if a["type"] == "paper" and
                            (include_unresolved or self.paper_state(n) == "retrieved"))
        graph.add_edges_from((u, v) for u, v, a in self.graph.edges(data=True) if a["relation"] == "PAPER_CITES_PAPER"
                             and u in graph and v in graph and self._citation_verified(u, v))
        return graph

    def neighbors(self, node_id: str, *, relation: str | None = None, direction: str = "both") -> list[str]:
        node_id = entity_id(node_id)
        self._index()
        if node_id not in self.graph:
            raise ValueError("Unknown graph entity")
        if direction not in ("in", "out", "both") or relation is not None and relation not in RELATIONS:
            raise ValueError("Invalid traversal direction or relation")
        result = set()
        if direction in ("out", "both"):
            result.update(v for u, v, a in self.graph.out_edges(node_id, data=True) if (relation is None or a["relation"] == relation)
                          and (a["relation"] != "PAPER_CITES_PAPER" or self._citation_verified(u, v)))
        if direction in ("in", "both"):
            result.update(u for u, v, a in self.graph.in_edges(node_id, data=True) if (relation is None or a["relation"] == relation)
                          and (a["relation"] != "PAPER_CITES_PAPER" or self._citation_verified(u, v)))
        return sorted(result)

    def neighborhood(self, seeds: Iterable[str], *, hops: int = 1, max_nodes: int = 50,
                     direction: str = "both", relations: Iterable[str] | None = None) -> dict:
        if type(hops) is not int or hops < 0 or type(max_nodes) is not int or max_nodes < 1:
            raise ValueError("Traversal needs nonnegative integer hops and a positive node cap")
        if direction not in ("in", "out", "both"):
            raise ValueError("Invalid traversal direction")
        allowed = set(RELATIONS if relations is None else [relations] if isinstance(relations, str) else relations)
        if allowed - RELATIONS.keys():
            raise ValueError("Unknown graph relation")
        seeds = [seeds] if isinstance(seeds, str) else seeds
        seeds = list(dict.fromkeys(entity_id(s) for s in seeds))
        if any(s not in self.graph for s in seeds):
            raise ValueError("Unknown neighborhood seed")
        scheduled = set(seeds[:max_nodes])
        queue = deque((s, 0) for s in seeds[:max_nodes])
        selected = set()
        while queue:
            node, depth = queue.popleft()
            selected.add(node)
            if depth < hops and len(scheduled) < max_nodes:
                neighbors = set(n for relation in allowed for n in self.neighbors(node, relation=relation, direction=direction))
                for neighbor in sorted(neighbors - scheduled):
                    scheduled.add(neighbor)
                    queue.append((neighbor, depth + 1))
                    if len(scheduled) == max_nodes:
                        break
        view = self.graph.subgraph(selected).copy()
        view.remove_edges_from([(u, v, k) for u, v, k in view.edges(keys=True) if k not in allowed or
                                k == "PAPER_CITES_PAPER" and not self._citation_verified(u, v)])
        return _node_link(view)

    def path(self, paper_id: str) -> list[str]:
        node, result = canonical_id(paper_id), []
        while node:
            if node in result:
                raise ValueError("Cycle in discovery paths")
            if node not in self.graph:
                if not result:
                    return []
                raise ValueError("Missing discovery parent")
            result.append(node)
            node = self.graph.nodes[node].get("parent")
        return list(reversed(result))

    def stats(self) -> dict:
        return {"nodes": self.graph.number_of_nodes(), "edges": self.graph.number_of_edges(),
                "paper_nodes": sum(a["type"] == "paper" for _, a in self.graph.nodes(data=True)),
                "resolved_papers": sum(a["type"] == "paper" and a.get("resolved", False) for _, a in self.graph.nodes(data=True))}

    def detailed_stats(self) -> dict:
        self._index()
        stats = self.stats()
        return {**stats, "node_types": dict(Counter(a["type"] for _, a in self.graph.nodes(data=True))),
                "relations": dict(Counter(a["relation"] for _, _, a in self.graph.edges(data=True))),
                "paper_states": dict(Counter(self.paper_state(n) for n, a in self.graph.nodes(data=True) if a["type"] == "paper")),
                "retired_entities": len(self.graph.graph.get("retired_entities", {})),
                "unverified_citations": sum(k == "PAPER_CITES_PAPER" and not self._citation_verified(u, v)
                                            for u, v, k in self.graph.edges(keys=True)),
                "unresolved_papers": stats["paper_nodes"] - stats["resolved_papers"],
                "max_discovery_depth": max((a["depth"] for _, a in self.graph.nodes(data=True) if a["type"] == "paper"), default=0)}

    def validate(self) -> None:
        """Validate structure and evidence against persisted observed snapshots."""
        schema = self.graph.graph.get("scholarly_schema", 0)
        if not self.graph.is_directed() or not self.graph.is_multigraph() or type(schema) is not int or schema not in (0, 1, 2):
            raise ValueError("Unsupported scholarly graph schema")
        # Validation also detects unsupported edits made through the raw graph.
        self._indexed_graph = None
        self._index()
        retired = self.graph.graph.get("retired_entities", {})
        if not isinstance(retired, dict):
            raise ValueError("Malformed historical entity archive")
        for node, attrs in retired.items():
            kind = attrs.get("type")
            if kind not in NODE_KINDS or canonical_id(node, NODE_KINDS[kind]) != node or node in self.graph:
                raise ValueError("Invalid historical entity identity")
            if kind == "paper" and (attrs.get("resolved") is not False or any(k in attrs for k in ("metadata", "retrieved_at"))):
                raise ValueError("Resolved papers cannot be retired as citation targets")
            if kind == "paper":
                parent = attrs.get("parent")
                if parent is not None and (not isinstance(parent, str) or parent not in self.graph or
                                           self.graph.nodes[parent].get("type") != "paper" or not self.graph.nodes[parent].get("resolved")):
                    raise ValueError("Historical citation target has an invalid discovery parent")
                if type(attrs.get("depth")) is not int or attrs["depth"] < 0:
                    raise ValueError("Historical citation target has an invalid discovery depth")
            if not isinstance(attrs.get("discoveries", []), list) or any(not isinstance(d, dict) or
                    not isinstance(d.get("action"), str) or type(d.get("depth")) is not int
                    for d in attrs.get("discoveries", [])):
                raise ValueError("Historical entity has invalid discovery records")
        papers = {}
        for node, attrs in self.graph.nodes(data=True):
            kind = attrs.get("type")
            if kind not in NODE_KINDS or canonical_id(node, NODE_KINDS[kind]) != node:
                raise ValueError("Graph contains an invalid typed entity")
            if kind == "paper":
                if type(attrs.get("resolved")) is not bool or type(attrs.get("depth")) is not int or attrs["depth"] < 0:
                    raise ValueError("Invalid paper resolution or discovery depth")
                if schema == 2 and attrs.get("status") != ("retrieved" if attrs["resolved"] else "unresolved"):
                    raise ValueError("Paper state disagrees with resolution")
                parent = attrs.get("parent")
                if parent and (parent not in self.graph or not self.graph.nodes[parent].get("resolved")):
                    raise ValueError("Invalid discovery parent")
                expected = self.graph.nodes[parent]["depth"] + 1 if parent else 0
                if attrs["depth"] != expected:
                    raise ValueError("Inconsistent discovery depths")
                if attrs.get("metadata"):
                    paper = _paper_copy(Paper.from_dict(attrs["metadata"]))
                    if not attrs["resolved"] or paper.id != node or paper.retrieved_at != attrs.get("retrieved_at"):
                        raise ValueError("Paper snapshot disagrees with graph identity")
                    papers[node] = paper
                elif attrs["resolved"] and schema in (1, 2):
                    raise ValueError("Resolved paper has no metadata snapshot")
                if not attrs["resolved"] and attrs.get("title"):
                    raise ValueError("Unresolved paper contains unsupported metadata")
                if schema == 2 and not attrs["resolved"] and any(k in attrs for k in ("metadata", "retrieved_at", "metadata_history")):
                    raise ValueError("Unresolved targets cannot contain retrieved metadata")
                for historical in attrs.get("metadata_history", []):
                    previous = _paper_copy(Paper.from_dict(historical["metadata"]))
                    if previous.id != node or previous.retrieved_at > attrs["retrieved_at"]:
                        raise ValueError("Invalid historical paper observation")
        for u, v, key, attrs in self.graph.edges(keys=True, data=True):
            if key not in RELATIONS or attrs.get("relation") != key or (self.graph.nodes[u]["type"], self.graph.nodes[v]["type"]) != RELATIONS[key]:
                raise ValueError("Unsupported scholarly relationship or direction")
            evidence = attrs.get("evidence")
            if not isinstance(evidence, list) or not evidence or len(set(evidence)) != len(evidence):
                raise ValueError("Relationship lacks unique supporting evidence")
            if key != "AUTHOR_AFFILIATED_WITH_INSTITUTION" and evidence != [u]:
                raise ValueError("Paper relationship has incorrect supporting work")
            observations = attrs.get("observations", {})
            if not isinstance(observations, dict) or schema in (1, 2) and set(observations) != set(evidence):
                raise ValueError("Relationship observations disagree with supporting works")
            if schema == 2 and attrs.get("source_field") != SOURCE_FIELDS[RELATION_FIELDS[key]]:
                raise ValueError("Relationship lacks its source metadata field")
            for owner in evidence:
                if owner not in self.graph or self.graph.nodes[owner].get("type") != "paper" or not self.graph.nodes[owner].get("resolved"):
                    raise ValueError("Relationship evidence is not an observed work")
                if key == "AUTHOR_AFFILIATED_WITH_INSTITUTION":
                    authorship = self.graph.get_edge_data(owner, u, "PAPER_WRITTEN_BY_AUTHOR") or {}
                    if owner not in authorship.get("evidence", []):
                        raise ValueError("Affiliation is not supported by authorship")
                if owner in papers:
                    field = RELATION_FIELDS[key]
                    expected_source = self.graph.nodes[owner].get("field_sources", {}).get(field, {
                        "retrieved_at": papers[owner].retrieved_at, "provenance": papers[owner].provenance})
                    if owner in observations and observations[owner] != expected_source:
                        raise ValueError("Relationship source disagrees with observed metadata")
                    if schema == 2 and key == "PAPER_CITES_PAPER" and expected_source.get("metadata_source") not in OPENALEX_SOURCES:
                        raise ValueError("Citation has no OpenAlex metadata source")
        for owner, paper in papers.items():
            if self._owners[owner] != _relationships(paper):
                raise ValueError("Graph relationships disagree with observed work metadata")

    def to_dict(self) -> dict:
        self.validate()
        return _node_link(self.graph)

    @classmethod
    def from_dict(cls, data: dict, *, _consume: bool = False) -> "ScholarlyGraph":
        """Read current edges and legacy NetworkX links snapshots safely."""
        try:
            if data.get("directed") is not True or data.get("multigraph") is not True:
                raise ValueError("Scholarly snapshots must be directed multigraphs")
            if "edges" not in data and "links" not in data:
                raise ValueError("Snapshot has no edge collection")
            graph = nx.MultiDiGraph(**deepcopy(data.get("graph", {})))
            for item in data["nodes"]:
                attrs = item if _consume else deepcopy(item)
                node = attrs.pop("id")
                if node in graph:
                    raise ValueError("Duplicate snapshot node")
                graph.add_node(node, **attrs)
            for item in data.get("edges", data.get("links", [])):
                attrs = item if _consume else deepcopy(item)
                u, v, key = attrs.pop("source"), attrs.pop("target"), attrs.pop("key")
                if u not in graph or v not in graph or graph.has_edge(u, v, key):
                    raise ValueError("Dangling or duplicate snapshot edge")
                graph.add_edge(u, v, key=key, **attrs)
            result = cls()
            result.graph = graph
            result.validate()
            return result
        except (AttributeError, KeyError, TypeError, nx.NetworkXError):
            raise ValueError("Malformed scholarly graph snapshot") from None

    def save(self, path: str | Path) -> None:
        """Stream a validated snapshot without duplicating the complete graph."""
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        def write_items(stream, items):
            first = True
            for item in items:
                if not first:
                    stream.write(",")
                json.dump(item, stream, ensure_ascii=False, allow_nan=False)
                first = False
        ordered = sorted(self.graph)
        def edges():
            for u in ordered:
                for v in sorted(self.graph.adj[u]):
                    for key, attrs in sorted(self.graph.adj[u][v].items()):
                        yield {**attrs, "evidence": sorted(attrs["evidence"]), "source": u, "target": v, "key": key}
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write('{"directed":true,"multigraph":true,"graph":')
            json.dump(dict(self.graph.graph), stream, ensure_ascii=False, allow_nan=False)
            stream.write(',"nodes":[')
            write_items(stream, ({**self.graph.nodes[n], "id": n} for n in ordered))
            stream.write('],"edges":[')
            write_items(stream, edges())
            stream.write("]}")
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "ScholarlyGraph":
        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream), _consume=True)
