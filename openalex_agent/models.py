"""Loss-aware normalization of OpenAlex metadata (no model-generated facts)."""

from dataclasses import asdict, dataclass, field
from copy import deepcopy
from datetime import datetime, timezone
import re
from typing import Any


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_id(value: str, kind: str = "W") -> str:
    """Normalize short IDs or official OpenAlex URLs, rejecting foreign URLs."""
    match = re.fullmatch(r"(?:https?://openalex\.org/)?([WATI]\d+)", value.strip() if isinstance(value, str) else "", re.I)
    if not match or not match[1].upper().startswith(kind):
        raise ValueError(f"Expected an OpenAlex {kind} identifier")
    return "https://openalex.org/" + match[1].upper()


def reconstruct_abstract(index: dict | None) -> str | None:
    """Reconstruct supplied token positions, ignoring malformed optional entries."""
    if not isinstance(index, dict):
        return None
    positions: dict[int, str] = {}
    for word, offsets in (index or {}).items():
        if not isinstance(word, str) or not isinstance(offsets, list):
            continue
        for offset in offsets:
            if isinstance(offset, int) and not isinstance(offset, bool) and 0 <= offset < 100000:
                positions[offset] = word
    return " ".join(positions[i] for i in sorted(positions)) or None


@dataclass
class Entity:
    id: str
    name: str | None = None


@dataclass
class Authorship:
    author: Entity | None
    institutions: list[Entity] = field(default_factory=list)
    raw_name: str | None = None


@dataclass
class Paper:
    id: str
    title: str | None = None
    doi: str | None = None
    year: int | None = None
    authorships: list[Authorship] = field(default_factory=list)
    topics: list[Entity] = field(default_factory=list)
    abstract: str | None = None
    references: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    citation_count: int | None = None
    source_url: str | None = None
    venue: str | None = None
    publication_date: str | None = None
    work_type: str | None = None
    biblio: dict = field(default_factory=dict)
    retrieved_at: str = field(default_factory=utcnow)
    provenance: str = "OpenAlex API"
    # Structured ingestion provenance is independent of the display label above.
    metadata_source: str | None = None
    # None denotes a legacy/explicit complete observation; [] denotes no fields.
    metadata_fields: list[str] | None = None

    @property
    def text(self) -> str:
        return " ".join([self.title or "", self.abstract or "", *(t.name or "" for t in self.topics)])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Paper":
        data = dict(data)
        if "metadata_source" not in data and data.get("provenance") == "OpenAlex API":
            # Preserve old normalized caches, explicitly distinguishing them
            # from records ingested with the current field-presence contract.
            data["metadata_source"] = "openalex_legacy"
        data["authorships"] = [Authorship(
            Entity(**a["author"]) if a.get("author") else None,
            [Entity(**i) for i in a.get("institutions", [])], a.get("raw_name"),
        ) for a in data.get("authorships", [])]
        data["topics"] = [Entity(**t) for t in data.get("topics", [])]
        return cls(**data)

    @classmethod
    def from_openalex(cls, raw: dict[str, Any], *, provenance: str = "OpenAlex API") -> "Paper":
        """Retain usable works when optional entities or relationships are incomplete."""
        if not isinstance(raw, dict):
            raise ValueError("OpenAlex work must be an object")

        def text(value) -> str | None:
            return value if isinstance(value, str) and value.strip() else None

        def items(value) -> list:
            return value if isinstance(value, list) else []

        def entity(item: dict | None, kind: str) -> Entity | None:
            if not isinstance(item, dict) or not item.get("id"):
                return None
            try:
                return Entity(canonical_id(item["id"], kind), text(item.get("display_name")))
            except ValueError:
                return None

        def identifiers(value, field: str) -> list[str]:
            result = []
            for item in items(value):
                try:
                    result.append(canonical_id(item))
                except ValueError:
                    # A malformed list is not authoritative evidence that
                    # previously known relationships have disappeared.
                    observed.discard(field)
                    continue
            return list(dict.fromkeys(result))

        authorships = []
        for a in items(raw.get("authorships")):
            if not isinstance(a, dict):
                continue
            institutions = [e for i in items(a.get("institutions")) if (e := entity(i, "I"))]
            author = a.get("author") if isinstance(a.get("author"), dict) else {}
            authorships.append(Authorship(entity(author, "A"), institutions,
                                         text(a.get("raw_author_name")) or text(author.get("display_name"))))
        location = raw.get("primary_location") if isinstance(raw.get("primary_location"), dict) else {}
        source = location.get("source") if isinstance(location.get("source"), dict) else {}
        year = raw.get("publication_year")
        citations = raw.get("cited_by_count")
        field_names = {
            "doi": "doi", "publication_year": "year", "authorships": "authorships", "topics": "topics",
            "abstract_inverted_index": "abstract", "referenced_works": "references", "related_works": "related",
            "cited_by_count": "citation_count", "publication_date": "publication_date", "type": "work_type", "biblio": "biblio",
        }
        observed = {field for source, field in field_names.items() if source in raw}
        for raw_field in ("authorships", "topics", "referenced_works", "related_works"):
            if raw_field in raw and not isinstance(raw[raw_field], list):
                observed.discard(field_names[raw_field])
        if "title" in raw or "display_name" in raw:
            observed.add("title")
        if "primary_location" in raw:
            observed.update(("source_url", "venue"))
        return cls(
            id=canonical_id(raw.get("id", "")), title=text(raw.get("title")) or text(raw.get("display_name")),
            doi=text(raw.get("doi")), year=year if type(year) is int and year > 0 else None, authorships=authorships,
            topics=[e for t in items(raw.get("topics")) if (e := entity(t, "T"))],
            abstract=reconstruct_abstract(raw.get("abstract_inverted_index")),
            references=identifiers(raw.get("referenced_works"), "references"), related=identifiers(raw.get("related_works"), "related"),
            citation_count=citations if type(citations) is int and citations >= 0 else None,
            source_url=text(location.get("landing_page_url")), venue=text(source.get("display_name")),
            publication_date=text(raw.get("publication_date")), work_type=text(raw.get("type")),
            biblio=raw.get("biblio") if isinstance(raw.get("biblio"), dict) else {}, provenance=provenance,
            metadata_source="openalex", metadata_fields=sorted(observed),
        )


OPENALEX_SOURCES = {"openalex", "openalex_legacy"}
MERGE_FIELDS = {"title", "doi", "year", "authorships", "topics", "abstract", "references", "related",
                "citation_count", "source_url", "venue", "publication_date", "work_type", "biblio"}


def merge_observation(paper: Paper, previous: dict | None, sources: dict) -> tuple[dict, dict]:
    """Merge normalized metadata consistently with or without graph memory."""
    incoming = paper.to_dict()
    merged = deepcopy(previous) if previous else dict(incoming)
    supplied = set(paper.metadata_fields) if paper.metadata_fields is not None else set(MERGE_FIELDS)
    if paper.metadata_source not in OPENALEX_SOURCES:
        supplied.discard("references")
    for field in MERGE_FIELDS:
        value = incoming[field]
        if field not in supplied or value is None:
            continue
        if field == "biblio" and previous:
            value = {**previous.get("biblio", {}), **{k: v for k, v in value.items() if v is not None}}
        merged[field] = value
        sources[field] = {"retrieved_at": paper.retrieved_at, "provenance": paper.provenance,
                          "metadata_source": paper.metadata_source}
    # Old snapshots did not record field presence or individual sources.
    for field in MERGE_FIELDS - sources.keys():
        if merged[field] is None or merged[field] == [] or merged[field] == {} or merged[field] == "":
            continue
        sources[field] = {"retrieved_at": (previous or incoming)["retrieved_at"],
                          "provenance": (previous or incoming)["provenance"],
                          "metadata_source": (previous or incoming).get("metadata_source", "openalex_legacy")}
    merged.update(id=paper.id, retrieved_at=paper.retrieved_at, provenance=paper.provenance,
                  metadata_source=paper.metadata_source, metadata_fields=sorted(MERGE_FIELDS))
    return merged, sources
