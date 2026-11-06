# Knowledge graph correctness

All keys normalize to canonical OpenAlex URLs. Repeated IDs update one node;
identical titles with different IDs remain distinct works.

| Paper state | Meaning |
| --- | --- |
| `unresolved` | An ID reported by retrieved metadata; no fabricated bibliographic record |
| `retrieved` | A normalized metadata observation exists; completeness/full text are not implied |
| `legacy_metadata_missing` | Compatibility state requiring hydration before verified citation analysis |

Promotion updates the same node and preserves discovery routes. Citation-only
observations cannot demote a retrieved paper. Citation edges point citing to
cited and require structured ingestion provenance plus the source record's
`referenced_works` IDs. Related works, model guesses, topics and title similarity
cannot create citation edges. Incoming citation expansion verifies that the
returned work actually references the anchor.

Presence-aware merges distinguish absent fields from explicitly supplied empty
lists. Missing/null data cannot erase known metadata; explicit corrections update
current relationships while prior observations/evidence remain in history. Shared
edge support survives another source's update. Freshness and shortest discovery
paths are separate; older metadata cannot roll back a newer observation.

Per-source edge indexes and discovery-child indexes support incremental updates.
Versioned snapshots validate typed entities, canonical IDs, edge evidence and
parents/depths. Atomic streaming saves avoid a second complete serialized copy;
loading and validation still operate in memory. NetworkX is not an out-of-core
OpenAlex-wide store; degree, unresolved targets and accumulated history determine
memory use.

The suite includes synthetic capacity/correctness checks. A configurable larger
engineering workload is available without downloading literature:

```bash
python -m openalex_agent.graph_benchmark --papers 10000 --references 20 --updates 1000 --output data/graph-validation
```

Generated snapshots may be large and belong outside Git. This workload does not
establish scholarly retrieval quality or real-corpus memory requirements.
