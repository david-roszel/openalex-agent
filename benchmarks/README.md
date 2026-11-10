# Public examples and development inputs

`example_questions.json`, `example_factual_questions.json` and `example_review_candidates.json` are synthetic regression inputs. Their39-record synthetic corpus is under tests/fixtures/synthetic-corpus. They contain no real human judgments.

`live_submission_v1/questions.json` is the fixed eight-question development/target-identity input used by the live research runner and its tests. It is not held-out evaluation material or general relevance annotation. Only that input is released here: no private campaign traces, target-resolution receipts, approvals, answer keys or recorded-attempt outputs are copied.

The separate tests/fixtures/development-corpus contains39 authentic historical metadata records for offline adapter diagnostics, with provenance. It is not the frozen30k corpus. Use your own reviewed inputs for empirical evaluation, retain independent judgments and keep sealed questions local. See ../docs/EVALUATION.md and ../docs/REPRODUCIBILITY.md.
