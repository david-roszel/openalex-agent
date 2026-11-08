# Public offline fixtures

`openalex-list.synthetic.json` simulates an OpenAlex response.
`synthetic-corpus/` contains 39 explicitly synthetic records plus an integrity
manifest/provenance file. Synthetic network/multi-relation/automation examples
exercise review and evidence-accounting mechanics. No record is a real paper,
research finding, private held-out question or independent human judgment.
Other test inputs are generated in temporary directories. The tokenizer asset
and its upstream license are packaged separately under openalex_agent/tokenizer_data.

`meta-context-failed-edits.json` contains synthetic context text and invalid
argument-key combinations for tool-contract regression. It is not a provider
trace. Exact historical Meta and eight-overflow CLM snapshots remain private.
The optional SDK probes mock HTTP and do not authorize hosted generation.
