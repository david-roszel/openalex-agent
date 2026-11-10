# Graph correctness and capacity

This refinement stays within Milestone 2. It changes graph construction,
metadata provenance/presence and persistence, not agent or LLM behavior.

## Identity and paper state

All graph keys use canonical OpenAlex URLs. Short IDs, HTTP/HTTPS URLs, letter
case and surrounding whitespace normalize before insertion. Different IDs with
identical titles remain different works; repeated IDs cannot duplicate nodes.

| State | Stored information | Transition |
| --- | --- | --- |
| `unresolved` | Canonical ID, discovery time/routes, best parent/depth; no bibliographic metadata | A verified citing work reports the target ID |
| `retrieved` | Normalized metadata snapshot, field provenance, acquisition time and historical observations | Metadata for that ID is supplied to `add` |
| `legacy_metadata_missing` | Compatibility state for older resolved nodes without their saved records | Hydrate metadata before citation analysis/expansion |

`retrieved` means a metadata record is available, not that full text exists or
every bibliographic field is complete. `resolved` remains as a compatibility
boolean. Promotion updates the existing node in place, preserving first discovery,
incoming edges and prior routes. A later citation-only encounter never demotes it.

Use `paper_state(id)` to inspect state and `paper(id)` to retrieve a copy of known
metadata. `citation_graph(include_unresolved=False)` excludes unresolved targets
when an analysis should consider only acquired records. CLI graph statistics
report paper states separately; unresolved targets do not count as retrieved
literature. Retired targets remain historically inspectable but are not counted
as active graph nodes.

## Citation provenance

`Paper.from_openalex` labels structured ingestion provenance as `openalex` and
records which normalized fields the response supplied. New citations require
this provenance and arise only from canonical IDs in `referenced_works`.
Related-work lists, similar titles, topics and model interpretations cannot
create citation edges. A free-form display label such as `OpenAlex API` on a
new manually constructed `Paper` does not authorize citations.

Edges point citing → cited and record `source_field=referenced_works`, supporting
work IDs and per-field acquisition/source information. Validation checks the
edge against the retained source record. Incoming citation expansion still
requires the returned work to explicitly contain the anchor in its references.
Unverified legacy structural citations stay stored for compatibility but are
excluded from citation traversal and analysis until metadata supports them.

Existing normalized API caches predating structured provenance are explicitly
tagged `openalex_legacy` on import. This trusts their original declared ingestion
contract; it is not cryptographic authentication. Production inputs must be
actual API responses or trusted OpenAlex caches. Privileged Python callers can
forge arbitrary objects, so the graph is not a security boundary against its
own caller. Synthetic fixtures deliberately simulate API responses and are
clearly marked in titles, provenance and engineering artifacts.

## Incremental updates without discarding prior information

For each work, the graph retains current accumulated metadata, the latest raw
observation digest, field-level sources and prior accepted snapshots.

- Fields absent from a partial response retain their previous values.
- Null/invalid optional fields do not clear known scalar metadata or lists.
- Explicit observed lists, including empty lists, can correct current edges;
  old metadata and its field provenance are archived first.
- Older observations cannot roll back newer metadata.
- Replaying the same observation or accumulated snapshot is idempotent.
- Updating one work removes only its superseded support; shared affiliations
  survive if another work still supports them.
- Retired unresolved targets and unused entities keep their first discovery,
  known names and source details in an archive. Rediscovery restores those
  details rather than creating a new identity or losing known names.

`paper_history(id)` and `historical_entity(id)` return copies of retained data.
History, current state and archives survive save/load. Historical discovery
routes are independent of the currently observed citation graph. The graph
preserves supported normalized fields; complete raw API responses remain the
retrieval cache's responsibility. List truncation and source omissions still
limit completeness; an absent current edge does not prove a relationship false.

## Complexity and persistence

Per-work edge ownership indexes target updates to affected relationships.
Discovery keys and evidence-membership sets make repeated insertion into a
popular citation target or shared affiliation constant time per observation,
rather than copying/scanning all previous supporters. The parent/children
index propagates shortened discovery paths. Serialization sorts evidence once,
instead of sorting it after every insertion.

Schema 2 persistence uses canonical node-link JSON with atomic replacement.
It streams individual records to disk, avoiding an entire copied export tree
and a second complete JSON string. JSON loading consumes decoded records without
deep-copying the whole tree again. Public `to_dict` still returns an isolated
copy for callers that need one; use `save` for large snapshots. Full validation,
save/load and staged manual expansion remain graph-wide operations. NetworkX
and JSON are in-memory tools, not an out-of-core database. Retained history
grows with accepted updates rather than being silently truncated.

## Executed 10,000-paper validation

```bash
.venv/bin/python -m openalex_agent.graph_benchmark --papers 10000 --references 20 --updates 1000 --output data/validation/graph-scale-10000
```

The actual run on this Ubuntu/Python 3.14.4 environment checked:

- 10,000 acquired synthetic paper records, each with 20 reference IDs;
- a shared heavily cited target and a shared affiliation supported by all papers;
- 1,000 partial updates without losing citations, abstracts or authors;
- canonical deduplication and promotion of 10 targets without adding nodes;
- bounded traversal, metadata/evidence validation and save/load round trip.

The final graph contained **30,010 nodes and 240,001 edges**, including 200,000
citations. Its 20,000 paper nodes comprised **10,010 retrieved records** after
promotion and **9,990 unresolved targets**. No unverified citations were found.

| Operation | Observed time |
| --- | ---: |
| Build 10,000 records | 2.78 s |
| Apply 1,000 partial updates | 0.14 s |
| Save validated snapshot | 10.53 s |
| Load and validate snapshot | 3.60 s |

Peak process RSS was about **990 MiB**; the uncompressed snapshot was about
**188 MiB**. Timings and memory are descriptive measurements of this synthetic
workload and machine, not universal limits. Real abstracts, references, degree
distributions and accumulated history change resource needs.

Measured artifacts are ignored under
`data/validation/graph-scale-10000/engineering-validation.json` and
`synthetic-graph.json`. They are **engineering evidence, not scholarly metadata,
relevance judgments or agent-performance findings**.

An automated regression also constructs 10,000 records with four references
each, applies updates, promotes targets and restores an 80,001-edge graph.
At the graph-hardening checkpoint, **139 tests passed**, including **78 graph-focused tests**.
See [current status](RESULTS_AND_LIMITATIONS.md) for subsequent milestones and the current suite count.
The existing cache of six genuine OpenAlex works also builds a validated
schema-2 graph with six retrieved papers and 108 unresolved targets, without
network requests. No larger real corpus or research results are claimed.
