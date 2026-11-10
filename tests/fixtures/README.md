# Offline test inputs

- `openalex-list.synthetic.json` is an explicitly synthetic HTTP-response fixture.
- `development-corpus/` contains an unchanged, checksummed copy of the 39-record
  OpenAlex metadata snapshot used by review-protocol regression tests. Its
  `fixture-provenance.json` identifies the original frozen source and hashes.
  These are real metadata/abstract records, not synthetic papers or independent
  relevance/factual ground truth. They must not be presented as research results.

The small copy makes the test suite independent of the ignored local `data/`
archive. The original frozen corpus has not been modified. Normalized observations
retain their original acquisition metadata. Source abstracts retain their source
attribution; this project does not grant rights to third-party source material.
The existing source URLs remain in the records. There is no licensed full text,
credential, private reviewer worksheet or hosted request log in that 39-record copy.

`clm-pending-overflows.dev.json` separately contains selected request/response
snapshots for the eight preserved November 6 development-pilot admission failures.
It includes original selected inputs, decisions, archive bodies and source hashes
for deterministic offline replay. It contains no authentication headers, API
keys, private reviewer annotations, or held-out questions. These real historical
model decisions are regression inputs, not newly generated hosted results or
independent relevance/grounding judgments. Tests write only to temporary workspaces
and do not require the local historical pilot directory.

Also track `benchmarks/*.json`, which contains proposed question sets, public
factual probes and the frozen-pilot receipt. Those files contain no independent
relevance grades. The packaged tokenizer ranks and accompanying MIT license under
`openalex_agent/tokenizer_data/` are necessary for network-free token measurement.
Other fixtures are generated inside pytest temporary directories.
