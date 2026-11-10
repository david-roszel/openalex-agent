# Reproducibility and released material

Current UI/CLI research shares the instrumented research adapter. Controlled benchmark execution preserves isolated method state, raw input/output snapshots, source/question/corpus/provider/prompt identities, budgets, failures and accounting. Git provenance records commit, dirty state and explicit unavailable statuses; source hashes still identify dirty worktrees. No private Git metadata or credential file is distributed.

There are three different public input classes:

1. `benchmarks/example_*.json` and `tests/fixtures/synthetic-corpus`: authored synthetic correctness examples, never independent human truth.
2. `tests/fixtures/development-corpus`:39 authentic historical OpenAlex metadata records with provenance/checksums; not the30k study corpus or an evaluation gold set.
3. `benchmarks/live_submission_v1/questions.json`: eight fixed development questions with predeclared publication identities required by the exploratory runner/tests. Not held-out questions, human relevance grades or successful results. Historical pending-resolution text remains versioned input, not a claim about today's API availability.

The aggregate file in `examples/research-status` describes previously audited outcomes. Raw empirical outputs, full corpus, private task/gold packages and human annotations are omitted; source-document hashes provide provenance, not independent reanalysis from aggregates. Do not claim fully reproducible empirical efficacy from this partial public data distribution.

The separate `versions/revision7` source is frozen; its118 pins match the original runtime3. Its public synthetic scoring/accounting/scheduler tests are portable without private inputs. Original real-corpus preparation/trajectory tests are not silently labeled public or replaced with favorable synthetic results. Historical builder path constants and spending limits remain frozen, not activated or made into approvals.

For your own evaluation: obtain appropriately sourced data, freeze questions and shared budgets before outcomes, record source/prompt/provider identities and original inputs, keep held-out questions separate, and complete independent blinded relevance/grounding review. The primary pooled relevance rubric is0–2; unjudged papers remain UNKNOWN. Generic review preparation accepts explicit question/probe/candidate/reserved paths. No approval or annotation is fabricated by a generated template.

Do not mutate frozen originals when scoring; copies/archival exports retain parent identities. Record actual evidence exposure, final-answer status and failure policy alongside retrieval scores. Scheduled papers/checkpoints are not necessarily delivered records. Trial identities are not guaranteed hosted sampling seeds; identical requests may vary.

`PUBLIC_MANIFEST.json` covers prepared publication files, excluding its own hash. Verify it with `python scripts/verify_manifest.py` before installing/running if a stable source identity is needed. A source edit requires a new manifest/revision; generated runtime outputs stay in ignored data/. New release code does not retroactively change archived experiment hashes or validate old outcomes.

Offline test commands use the inherited socket/write guard and mock providers. CI declares multiple Python versions; only locally executed versions are reported as observed passes. Tokens are measured with the packaged study tokenizer where configured, not assumed identical to provider billing/framing.
