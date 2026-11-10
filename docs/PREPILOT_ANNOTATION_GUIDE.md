> Historical worksheet guidance. This public release uses explicit user-selected or synthetic example inputs; references below to a fixed three-question/39-paper worksheet are examples, not a released human-reviewed dataset. The0–2 rubric and independent-review requirements remain unchanged.

# Development annotation instructions

The packet is **not ground truth yet**. All grades, verified answers and reviewer identities are blank. Source-selection notes and provisional adequacy judgments are assistant preparation, not independent annotation. No agent responses or held-out evaluation results were used to create it.

## Exact files

- `data/prepilot-development-v1/relevance-worksheet.csv` (generated locally; not distributed).
- `data/prepilot-development-v1/second-review-sample.csv` (generated locally; not distributed), eight per research question, selected before labels.
- `data/prepilot-development-v1/evidence-reviewer-worksheet.csv` (generated locally; not distributed), without suggested adequacy labels.
- `data/prepilot-development-v1/evidence-adequacy-worksheet.csv` (generated locally; not distributed), with clearly marked provisional source-inspection notes. Keep these suggestions away from first-pass reviewers until their independent sheets are submitted.
- `data/prepilot-development-v1/questions.json` (generated locally; not distributed), `data/prepilot-development-v1/factual-questions.json` (generated locally; not distributed), `data/prepilot-development-v1/review-manifest.json` (generated locally; not distributed). Individual source files are under `data/prepilot-development-v1/sources/` and include the saved OpenAlex title/abstract and acquisition provenance.

Reviewers should use the relevance/factual reviewer sheets and actual source text, independently of agent outputs and each other's judgments. Give each human an anonymized reviewer identifier. `assessor` must identify the actual reviewer, not an invented person or a model/provider name. Keep the reviewer-to-person mapping privately if needed.

## Relevance review

One reviewer assesses all 117 question–paper pairs. A second independently reviews the 24-row overlap; neither sees the other's grades before submission. The candidate pool is the same complete 39-paper snapshot for each question, shuffled without retrieval scores/method identifiers. Inclusion does not imply relevance.

Fill these columns:

| Field | Required entry |
| --- | --- |
| `grade` | `0`: does not address the question; `1`: partly addresses it or supplies useful background; `2`: directly addresses it. Use this development scale, not grade 3. |
| `supporting_passage` | Exact text from the provided abstract when available; leave blank if no passage supports the judgment and explain why. |
| `explanation` | Why this paper receives this grade for this particular question. Required even for grade 0. |
| `judgment_provenance` | Human review method, material inspected and date; e.g. “independent abstract review, YYYY-MM-DD”. Disclose any assistance. |
| `assessor` | Actual human's anonymized identifier. |
| `independent_human_review` | `true` only after an actual independent human review. |
| `uncertainty` / `disagreement` | `true` or `false`. `unreviewed` is the initial state, not a completed flag. |
| `adjudication_note` | Explanation of any resolved conflict or revised judgment. |

Do not infer results from titles or count a missing abstract as proof of irrelevance. Mark uncertainty if the supplied material cannot support a reliable relevance judgment. Acquire/review additional evidence in a separately versioned packet if necessary; do not silently change the shared corpus or force uncertain cases to grade 0.

Retain both original reviewer sheets unchanged. Compare the overlap, record agreement/disagreements, and adjudicate disagreements using documented discussion or another reviewer. Transfer adjudicated grades into the main relevance worksheet. A final `disagreement=false` after a conflict requires an adjudication note; preserve original flags/grades in the reviewer originals. Do not clear uncertainty merely to pass validation.

For the existing retrieval evaluator, binary relevance means **grade > 0**; nDCG uses graded gains `2^grade-1`, giving partial/direct grades gains 1/3. If all 39 papers are judged for every question, recall is exhaustive only for this candidate corpus, not all OpenAlex literature.

## Factual evidence adequacy

Two humans independently review all three factual probes. Use the factual reviewer worksheet first; the coordinator transfers final adjudicated review fields into the adequacy worksheet. The `candidate_passage` is available source text, not a verified answer. The third probe is intentionally a source-limited question; absence of a quantified result must be checked, not filled with an estimate.

Choose `reviewed_adequacy`:

- `sufficient`: available source text establishes the complete requested answer.
- `partially_sufficient`: establishes only part; explicitly state the remaining gap. Partial support is not a full verified answer.
- `insufficient`: cannot establish the requested answer from the available evidence.

For sufficient or partially sufficient material, provide `verified_answer`, canonical `supporting_work_id`, exact `supporting_passage`, `source_type=abstract`, and `evidence_location=sources/W<ID>.json#abstract`. The passage must actually occur in the named source abstract. Describe partial scope in the answer/adjudication note. Fill provenance, actual reviewer ID, uncertainty/disagreement and independent-review fields as above. Set `human_verification_status=verified` only after verification/adjudication.

For an insufficient question, leave `verified_answer` blank rather than inventing it; explain the gap in `adjudication_note`. Record human verification of that inadequacy. Source titles and links do not establish detailed experimental results. No full text was acquired in this packet; if full text is needed, curate/hash it in a new version and review its availability to agents separately. Do not label a URL as reviewed full text.

The verified reference material stays private to evaluators. It is never added to `questions.json`, `factual-questions.json`, the shared corpus, planner context or graph. The public factual manifest accepts only `query_id`, `question_id` and `question`; extra answer/label fields are rejected before model construction.

## Validate and import completed review

Only the two coordinator worksheets are editable inputs to this checker; source/public-question files are checksum protected. Original reviewer sheets and any disagreement records should be retained separately.

```bash
.venv/bin/python -m openalex_agent.development_review \
  --check data/prepilot-development-v1
```

The current expected result is **UNKNOWN**, with 120 pending entries. Passing structural checks later cannot prove that a reviewer was actually independent; the coordinator must retain the review records and attestations and approve the study protocol.

After all relevance rows are adjudicated and validation passes, reuse the existing importer:

```bash
.venv/bin/python main.py judgments-import \
  data/prepilot-development-v1/relevance-worksheet.csv \
  --scope exhaustive --output data/prepilot-development-v1/relevance-adjudicated.json
```

Do not run that import on blank or unreviewed sheets. It is not an instruction to conduct a pilot.

## Later factual-response judgments

After a separately authorized pilot, use the existing `claim-pool` / `score-research` workflow. With the fixed protocol configured, the pool contains only the single terminal response for each factual question; optional planning claims remain trace diagnostics, not extra scoring opportunities. The pool hides architecture identity.

Compare each answer with the independently verified reference and named source:

- `judgment`: `supported`, `unsupported`, `contradicted`, or `abstain`.
- `factually_correct`: `true` only if the answer correctly addresses the requested fact, including all requested material parts; `false` for an incorrect/nonanswering attempted answer; blank if truth cannot be established. Do not confuse unsupported with false.
- `citation_support_correct`: independently check the cited work/passage, not just ID validity. A correct answer with an unrelated citation can be factually correct but unsupported and wrongly attributed.
- `appropriate`: for abstentions, judge whether the evidence available to the agent could establish the answer. Global corpus answerability and actual observed evidence are separate; record uncertainty when access cannot be established.
- `assessor`: the actual reviewing human; retain originals and adjudicate contradictions.

IDs, observed/access flags, quote matches, and semantic support have separate meanings. Do not infer semantic support automatically from valid IDs or treat corpus membership as proof of access. Use recorded execution provenance and final request snapshots to inspect source availability.

The pool includes `available_evidence_context`, which is untrusted model-visible
text, not verified truth. Identical abstention statements in different evidence
contexts have different assessment IDs, so an appropriate abstention in one run
cannot automatically label another run's abstention appropriate. Method names
are hidden, but context structure may still reveal hints about an intervention;
apply the same rubric and document that limitation.

Each method/trial has **three declared factual opportunities**. Primary answer accuracy is correct answers / 3; grounded task completion is correct, supported, observed/access-verified answers / 3. Report answer and response coverage, missing/malformed counts, abstentions and unknown judgments. Accuracy among attempted questions is secondary and must always show its denominator/coverage. Appropriate abstention remains separate from a correct factual answer; no silent perfect-accuracy score for empty output. Longer answers or repeated claims get no extra credit.

If the source-limited control is confirmed unanswerable, maximum answer completion is 2/3 for this packet; appropriate-abstention behavior is reported separately. Do not change that control after seeing which architecture benefits. The pilot's purpose is interpretable reliability evidence, not a final performance claim.
