# Synthetic public examples

Only `example_questions.json`, `example_factual_questions.json` and
`example_review_candidates.json` are distributed. They are freshly authored
synthetic regression inputs, not development/held-out research benchmarks or
independently reviewed scientific ground truth. They contain no private questions
or human annotations. Their 39-record corpus is under tests/fixtures/synthetic-corpus.

Bring your own reviewed corpus/questions/annotations for empirical evaluation.
Keep reserved questions and identities local under ignored data/. See
[the evaluation methodology](../docs/EVALUATION.md).
