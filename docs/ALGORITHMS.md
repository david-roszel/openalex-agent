# Algorithms and information boundaries

`openalex.py` implements paced HTTP, retries, caching and usage accounting;
`storage.py` stores observations and checkpoints; `corpus.py` exposes checksummed
fixed snapshots with sparse TF-IDF search and reference/citation/topic lookups.
Live OpenAlex ranking and local TF-IDF ranking are different retrieval settings
and must not be combined as equivalent experimental conditions.

A candidate corpus is the indexed collection. Observed/inspected records have
actually been returned to the agent. Unresolved citation nodes are frontier IDs,
not observed papers. Only the search service sees the full fixed collection; no
full-corpus graph or hidden relevance labels are loaded into the planner.

The shared final ranker scores observed title/abstract/topic text against the
original question using nonnegative hashed word/bigram cosine similarity with
stable ID tie-breaking. Scores are not calibrated relevance probabilities.
Sharing ranking isolates discovery/planning effects from reranking effects.

Graph priority combines query relevance, observed novelty, personalized graph
importance, topic coverage and a coarse acquisition-cost prior. These are
configurable heuristics, not optimized or empirically established weights.
Unknown-target estimates are explicitly proxies. Ablations disable components
without silently renormalizing the remaining design. The graph changes planning
information and candidate proposals; it does not change the shared final ranker.

The controller validates schemas, acquired IDs, repetitions, depth, relevance
and budgets. All controlled agent cells retain the same acquisition tool surface.
Invalid hosted responses use logged, validated deterministic fallback. Do not
count a fallback trajectory as a successful hosted-policy observation.

Pending-action checkpoints preserve partial runs. New observations stage
metadata, graph state and trace records before committing. Existing run history,
provider fingerprints and budgets are checked on resume. Actual external cost
can remain uncertain after a process interruption; retain errors and failures.

Relevance labels belong only to evaluation. Report unknown-label coverage and
bounds instead of silently treating unknown papers as irrelevant. Factual support
requires independently supplied judgments; citation existence or lexical overlap
alone does not establish truth. See [evaluation](EVALUATION.md).
