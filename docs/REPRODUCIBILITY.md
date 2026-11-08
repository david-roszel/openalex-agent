# Reproducibility and released data

The CLI and dashboard share `BenchmarkSpec` and `run_experiments`. Each method
receives isolated run state. Manifests preserve source copies/hashes, corpus and
question checksums, budgets, model/provider settings, trial IDs, recorded seeds,
raw inputs/outputs, errors, timing and accounting. Git provenance records commit,
dirty state and explicit unavailable statuses; source snapshots are still needed
for dirty working trees. No private Git metadata is included in the export.

Recorded trial seeds do not guarantee hosted sampling reproducibility and are
not transmitted as an unsupported provider parameter. Identical requests can
produce different responses. Repetition is necessary for uncertainty assessment.

`benchmarks/example_*.json` and `tests/fixtures/synthetic-corpus/` are authored
synthetic correctness examples. Fixture checksums identify the exact input bytes;
there is no real OpenAlex corpus, private benchmark, human annotation, API log or
past hosted output in this release. Synthetic known outcomes are not empirical
human judgments. Tokenizer ranks retain their upstream license and attribution.

To use your own snapshot, acquire data with the corpus CLI, document query/date
bias and missing citation targets, then freeze it into a new `data/` directory.
Never overwrite a frozen source. Keep licensed/source provenance and permissions
with any separately distributed data. Frozen snapshot utilities remain available
for user-owned artifacts; no private snapshot receipt is supplied.

Build a blinded relevance pool with `pool`, import adjudicated CSV labels with
`judgments-import`, and apply existing `score-experiment` or factual scorers to
copies of saved runs. Final metrics require valid independent labels, not model
guesses. Dashboard assessment copies preserve their parent; lower-level CLI
scoring can update its chosen output, so do not point it at immutable originals.

`PUBLIC_MANIFEST.json` covers released files; it excludes its own checksum to
avoid a circular hash. Re-run the public tests and publication checker after any
change. Generated runtime files belong under ignored `data/`, and should not be
committed with source. Public experiment archival is a separate, explicitly
approved process with source/data permission review.

Verify the released source bytes with `python scripts/verify_manifest.py`. The
manifest excludes its own hash. Runtime outputs and later intentional edits are
outside that immutable export inventory; regenerate a new release manifest after
approved source changes. Integrity verification does not establish research validity.

Legacy CLM corrections carry `controlled-context-v2-capacity-stop`. New pinned
experiments must record this identity and updated source hashes. Do not combine
pre-correction fallbacks with corrected behavior as one treatment. Historical
request replays and frozen experiment artifacts remain private; public schema
fixtures contain synthetic context text and argument shapes only.
