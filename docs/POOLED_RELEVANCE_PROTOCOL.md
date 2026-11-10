# Pooled relevance protocol v1 — implementation available, scientific approval pending

Protocol identifier: `pooled-two-stage-v1`. Scope: Track-A frozen-corpus retrieval.
This is a separate, explicitly approved alternative to exhaustive annotation;
it does not mark the legacy dashboard gate ready. Current preparation materials
are unreviewed. No hosted experiment or scoring of actual human judgments has
been authorized or executed.

The primary study is fixed at `standard`, `llm`, `llm_graph`, `clm`, and
`clm_graph`. Official Meta CLM remains exploratory. The separately versioned
`five-method-development-collection-v1` adds `prepare-development`,
`check-development` and `collect-development` to the same CLI, using DEV inputs,
the benchmark runner and Gemini spending ledger. See
[development collection](FIVE_METHOD_DEVELOPMENT.md). The existing final
`collect` command and its held-out restrictions and approvals are unchanged.

## Frozen policy

The policy is the `POLICY` object in `openalex_agent/pooled_review.py`. Its digest
must be approved before collection. Any change creates new approvals and pools.

For each question, union and deduplicate by canonical OpenAlex work ID:

- Top 20 from the existing frozen-corpus lexical search.
- Up to five distinct references and five distinct incoming neighbors from the
  first three lexical anchors, using only frozen metadata.
- Five deterministic hash-selected background works. These are **unjudged**,
  not known negatives.
- After all complete experimental outputs are frozen: top 20 from **every**
  method/budget/trial, plus up to five deterministically sampled inspected works
  per cell. Keep every declared architecture; do not select successful winners.

All known/unknown identities remain within the approved corpus. The sampling
seed is `pool-blind-v1`; no relevance grades or architecture outcomes determine
pooling. Pooling stage and exact output-freeze hash are recorded. A pre-output
preparation pool cannot be passed to the final scorer. Corpus publication-identity
collisions require independent resolution before final protocol pins; automatic
DOI/title merging is not performed. Candidate provenance remains in a separate
coordinator file, never in reviewer worksheets.

Precision@10 uses relevance grades ≥1. nDCG gains are `2**grade - 1`. UNKNOWN
means unreviewed or insufficient evidence; it is never grade 0. This preserves
the current study's relevant threshold and gain rule with the requested 0–2 scale.

## Approval and freezing stages

`collect` is the new gated final Track-A entry point. It accepts public question
and corpus paths, budgets and inference ceilings, but no labels or hidden truth
path. It requires 10–20 held-out questions, three trials, independent approved
question/taxonomy/source-adequacy review, exact source/provider/settings/policy
pins, held-out independence attestation, explicit paid-inference authorization,
positive configured prices and an adequate conservative request-cost reservation.
An unreviewed worksheet, zero/unconfigured prices, false approval or wrong pin
stops before model construction. This implementation does not authorize a run.
Development hosted reliability runs still need their own approved protocol;
use of a lower-level CLI is not an exception to the research approval policy.

The native controlled-five runner preserves its schedule, code snapshots, inputs,
requests, context records, outputs, failures and operational statistics. No
relevance judgments are passed. Incomplete schedules cannot be frozen for scoring;
preserve and report the interruption, then approve a complete replacement version
or a missingness amendment. Do not delete failed arms or silently score only
successful cells. Label-free native metrics are not scientific quality findings.

`freeze` requires distinct coordinator approval pinning `experiment.json`,
`metrics.csv`, the SHA-256 inventory digest of the **entire raw output tree**,
and the pool policy. It copies raw files unchanged and writes normalized ranking
projections, result checksums and an `unscored-freeze.json` receipt. Absolute
legacy run paths are preserved as provenance; subsequent scoring reads the
relative, frozen ranking projection. Escaping run directories, symbolic links,
noncanonical/out-of-corpus identities and duplicate rankings are rejected.

`score` requires a third approval that pins the output freeze, complete pool
manifest, annotation version and policy. It writes a fresh assessment directory,
never into original/frozen experimental output. No new inference or retrieval
occurs. Scoring cannot silently adopt a later annotation or corpus version.
Approvals and reviewer attestations are administrative records, not cryptographic
proof of a person's independence. Only an authorized human may complete them.

## Reviewer instructions

Send only the relevant `reviewer-A/` or `reviewer-B/` worksheet and this rubric
to each reviewer. Do not send `coordinator/`, raw runs, method names, ranks,
pooling origins or the other reviewer's grades. Stable item IDs are hashes of
protocol/question/work identities, shuffled independently of origins and grades.
Source accessibility/connectivity and factual worksheets are separate review tasks.

Read the research question and available title/abstract. Grade:

| Grade | Meaning |
| --- | --- |
| 0 | Irrelevant to the requested question/scope |
| 1 | Partially relevant, useful background or limited direct evidence |
| 2 | Directly relevant to the requested question/scope |
| UNKNOWN | Insufficient available evidence, uncertain scope or not reviewed |

Do not infer irrelevance from missing detail. Do not invent source findings from
a title or citation count. Keep the supplied question/source identity and text
unchanged. For a known grade, enter a named reviewer ID and an explanation.
Supporting passages, if supplied, must be verbatim substrings of the available
source. Mark inaccessible/insufficient evidence UNKNOWN and explain the gap.
Do not use model-generated suggestions as independent grades.

Make a copy to edit; initial templates are immutable/checksummed. Complete the
separate human attestation (`reviewer_id`, `reviewed_at`, independent human
review and not-model-generated booleans). No attestation is prefilled as true.

Primary reviewer A covers the whole pool. Reviewer B independently covers
`ceil(20% × question pool size)` items within **each question**, chosen by the
fixed hash order, with a minimum of one. This is question-stratified, not
post-grade sampling. Both roles must use distinct reviewer identities. If a
single reviewer is all that is available, report the limitation and amend the
protocol; do not impersonate a second reviewer.

## Versioned imports, disagreements and adjudication

`import` copies the submitted worksheet and attestation into a new version and
records checksums, role, reviewer, pool pin and all assigned identities. Missing
rows/blank grades become UNKNOWN, never zero. Changed question/source text,
unassigned or duplicate row IDs, invalid grades, missing human attestations and
quotes outside the supplied source are rejected.

`resolve` requires both imports. Single-reviewed items use A's grade. Double-
reviewed items require two equal known grades; disagreement or either unknown
leaves the final grade UNKNOWN. It preserves both originals, grade-pair records,
counts of paired-known/disputed items, raw agreement and a coordinator-only
`adjudication-needed.csv`. Agreement is descriptive; no reliability claim is
made from a selectively easy subset.

A separate adjudicator may resolve disagreements/unknowns, recording a final
0/1/2 grade, reason, supporting passage if available and an independent human
attestation. The adjudicator cannot silently overwrite already agreed known
grades. Keep the prior unresolved annotation version and record a fresh resolved
version. `labels.json` contains known judgments and a separate `unknown_pairs`
list. `annotation-manifest.json` pins the submissions and adjudication records.
Actual independence, expertise and reason quality still need human oversight.

## Coverage, metrics and pooling limitations

Every score reports returned/judged/unjudged top-ten counts, top-ten coverage and
whole-pool coverage. Precision@10 is withheld if **any returned top-ten item is
UNKNOWN**. A short ranking counts unreturned slots as retrieval misses in the
fixed denominator ten; it does not fabricate grades for returned unknown works.
Report short rankings explicitly. Unknown top-ten items yield precision bounds,
not an ordinary Precision@10 score.

Pooled nDCG@10 is withheld unless both the evaluated top ten and **all candidates
in that question's frozen pool** have adjudicated known labels and the ideal
DCG is positive. A fully irrelevant judged pool gives undefined nDCG, not a
perfect score. All reported nDCG values are **relative to the judged pool**.
There is no full-corpus recall or exhaustive-corpus ideal ranking. Unpooled
relevant papers may exist even with 100% top-ten/pool coverage. Shared pools
reduce but cannot eliminate pooling bias, especially for systems absent from
the pool. Adding Meta later requires new predeclared pools and approvals.

The scorer emits all existing 2×2 question-level descriptive contrasts for both
metrics, including unknown/excluded-question records. It does not claim adjusted
significance. See the statistical policy in [the final plan](RESEARCH_EVALUATION_PROTOCOL.md).
Do not discard a question merely because one architecture lacks labels or fails.
The coordinator may approve additional blinded annotation rounds, pinning the
reason, all affected ranks and revised annotation versions before rescoring;
experimental inputs and outputs remain fixed.

## Current materials and annotation burden

`data/evaluation-readiness-2026-11-06-v1/development-review/` contains 24 DEV
questions, 765 primary pairs and 159 second-review pairs. `validation-review/`
contains 24 VALIDATION questions, 762 primary pairs and 159 second-review pairs.
These are genuine source materials from the 30,000-work snapshot, with blank
human decisions. The prepared pools have no experimental-output contributions
and therefore are **preparation only**.

Each contains question review, source access/connectivity, factual evidence and
adjudication templates, individual blinded review sheets, separate candidate
provenance, immutable hashes and a false human-attestation template. Final
held-out selection remains a sealed coordinator duty; no held-out text was
exported for this task.

For 18 questions × five methods × three repetitions × one budget, top-20 outputs
contribute at most 5,400 rank slots before deduplication; sampled inspections add
at most 1,350 and fixed baseline sources at most 630, giving a loose 7,380-pair
ceiling. The union is usually smaller but its size is UNKNOWN until outputs
exist. Double review adds approximately 20%, with per-question ceiling rounding,
plus adjudication. At two minutes per judgment, the loose ceiling exceeds 295
person-hours including double review. Even the 924 development assignments
would take about 31 person-hours at that assumed pace. Neither is a measured
annotation speed. Approve staffing/scope before collection; pooling is practical
relative to 540,000 exhaustive pairs but is not automatically finishable by one
person in four days. Reduce scope/pool depth only through a prospective protocol
amendment, never after favorable results are seen.

## Commands

From the project root, all outputs must be fresh. The saved false approval
examples and collection specification are in
`data/evaluation-readiness-2026-11-06-v1/protocol-preparation-v3/`.
Do not change an approval to true unless the authorized coordinator has actually
reviewed and approved the exact pins and costs.

```bash
# Offline preparation; currently no final human labels.
.venv/bin/python scripts/pooled_evaluation.py prepare \
  --corpus data/corpus/multidisciplinary-2026-11-06-v1 \
  --questions data/final-dataset-readiness-2026-11-06/integrated/development/questions.json \
  --output data/NEW-dev-pool

# ONLY after independent protocol/model/cost approval:
.venv/bin/python scripts/pooled_evaluation.py collect \
  --spec APPROVED_SPEC.json --question-review REVIEWED_QUESTIONS.csv \
  --approval APPROVED_COLLECTION.json
.venv/bin/python scripts/pooled_evaluation.py freeze \
  --experiment RAW_EXPERIMENT --approval APPROVED_FREEZE.json --output FROZEN_OUTPUTS
.venv/bin/python scripts/pooled_evaluation.py prepare \
  --corpus data/corpus/multidisciplinary-2026-11-06-v1 \
  --questions APPROVED_QUESTIONS.json --frozen-outputs FROZEN_OUTPUTS \
  --heldout-approval APPROVED_RELEASE.json --output FINAL_POOL

# Each reviewer submits a separate edited COPY and human attestation.
.venv/bin/python scripts/pooled_evaluation.py import --pool FINAL_POOL \
  --worksheet REVIEW_A.csv --attestation HUMAN_A.json --reviewer REVIEWER_A \
  --role A --output IMPORT_A_V1
.venv/bin/python scripts/pooled_evaluation.py import --pool FINAL_POOL \
  --worksheet REVIEW_B.csv --attestation HUMAN_B.json --reviewer REVIEWER_B \
  --role B --output IMPORT_B_V1
.venv/bin/python scripts/pooled_evaluation.py resolve --pool FINAL_POOL \
  --reviews IMPORT_A_V1 IMPORT_B_V1 --output UNRESOLVED_V1
.venv/bin/python scripts/pooled_evaluation.py resolve --pool FINAL_POOL \
  --reviews IMPORT_A_V1 IMPORT_B_V1 --adjudication ADJUDICATED.csv \
  --attestation HUMAN_ADJUDICATOR.json --adjudicator REVIEWER_C --output LABELS_V2
.venv/bin/python scripts/pooled_evaluation.py score --frozen-outputs FROZEN_OUTPUTS \
  --pool FINAL_POOL --annotations LABELS_V2 --approval APPROVED_SCORING.json \
  --output SCORED_COPY_V2
```

The uppercase paths are deliberate placeholders, not completed approvals or
runnable claims of readiness. Legacy claim/factual scoring remains a separate
independently reviewed Track-C workflow; pooled relevance does not certify truth.
