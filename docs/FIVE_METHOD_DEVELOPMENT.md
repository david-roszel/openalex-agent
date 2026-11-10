# Five-method development collection

Collection version: **`five-method-development-collection-v1`**. Review/scoring
remains **`pooled-two-stage-v1`**. No scientific findings or paid authorization
follow from successful configuration checks.

The primary methods are `standard`, `llm`, `llm_graph`, `clm`, `clm_graph`.
Official Meta CLM remains exploratory. The existing final `collect` command
still requires 10–20 held-out questions, three repetitions, independent review
and its separate approval. No final restriction is bypassed.

## Prepare and check offline

From the project root, substitute your Google Cloud project ID. Preparation
needs no credentials, SDK execution, API requests, human labels or approval file.
Choose a fresh output directory:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/pooled_evaluation.py prepare-development \
  --corpus data/corpus/multidisciplinary-2026-11-06-v1 \
  --questions data/gemini-seven-development-pilot-2026-11-06-v1/proposal/questions.json \
  --project YOUR_GOOGLE_CLOUD_PROJECT \
  --completion-guidance research-completion-guidance-v1 \
  --action-budgets 3 --trials 1 --max-llm-calls 6 \
  --output data/five-method-development-v1/proposal

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/pooled_evaluation.py check-development \
  --spec data/five-method-development-v1/proposal/spec.json
```

Preparation copies public DEV questions and writes a pinned spec and blank
`question-review.template.csv`. Check output separates valid configuration from
independent review readiness and lists the exact pins a later coordinator must
approve. An UNKNOWN template is not completed review. Make a copy for the human
reviewer; retain the template unchanged. Recheck with
`--question-review PATH_TO_COMPLETED_REVIEW.csv` before requesting authorization.
Software checks identities/attestations, not a person's actual independence.

To reserve the existing factual-response opportunity, add
`--factual-questions PATH_TO_PUBLIC_TASKS.json` during preparation. Supply exactly
one object per research question with only `query_id`, `question_id`, `question`.
Verified answers and annotations are rejected. Adding tasks later requires a new
proposal. Without tasks, optional planning claims do not give equal answer
opportunities or comparable task accuracy.

## Parity, identities and approval

All four agents share the corpus, provider, resource ceilings and lexical ranker.
Edits/recovery consume existing calls. Each declared factual response reserves
one of the same total calls. `standard` performs one search, makes zero LLM
calls and emits no fabricated factual response. Realized compute is measured,
not forced equal. Shared guidance is optional (`none` or
`research-completion-guidance-v1`) and identical across all four hosted methods.
The required CLM policy is `controlled-context-v2-capacity-stop`.

Spec/check pins identify question/factual-task bytes, corpus and manifest,
completed question-review bytes, exact source/scripts, provider/model/profile,
prompt templates/contracts/guidance, budgets/repeats/seeds and pooled policy.
Git availability/dirty status and native source snapshots remain recorded.
Any source/input/prompt/policy change requires new preparation and approval.

Paid authorization uses the existing pooled approval format: protocol
`pooled-two-stage-v1`, stage `collect-development`, exact check pins, named
approver/time, `approved=true`, `development_only=true`,
`paid_inference_authorized=true`, `final_evaluation_authorized=false`.
**No approval file is generated.** A final-stage approval cannot authorize DEV,
and a DEV approval cannot authorize held-out collection. Only after separate
human authorization:

```bash
.venv/bin/python scripts/pooled_evaluation.py collect-development \
  --spec PATH_TO_SPEC.json --question-review PATH_TO_COMPLETED_REVIEW.csv \
  --approval PATH_TO_EXPLICIT_AUTHORIZATION.json
```

## Safeguards and native outputs

The pinned Gemini ADC profile remains Gemini 3.8 Flash, global, LOW thinking,
SDK 3.0.0, 4,096 output tokens and 120-second timeout. Study ceilings remain
4,096 active / 8,192 full-input tokens, 16,000 characters and 100,000 total
inference tokens per agent run. DEV preparation supports 1–20 questions, 1–3
repetitions, action budgets up to 20 and at most 20 total calls per agent run.

The existing Gemini ledger reserves $2.13072 before each request and settles
against reported gross usage. Unknown usage retains its reservation and stops
immediately; confirmed spending at $3 requires review; retained spending is
capped at $6. Maximum hosted calls are four methods × questions × repetitions ×
action-budget conditions × per-run call ceiling. No retries or promotional-credit
assumptions are added. Runtime is bounded at two hours. These local safeguards
are not a provider billing-account cap or permission to spend.

Existing output directories are rejected. Native partial runs, errors, ledger
and status receipts survive stops; there is no automatic resume. Global accounting
stops escape acquisition fallback. The benchmark tree includes the copied ledger
and identity receipt so the existing raw-tree freeze preserves them.

Use the same separately approved `freeze`, output-inclusive `prepare`, `import`,
`resolve` and `score` stages. Preparation pools cannot replace output-inclusive
pools; incomplete schedules cannot be frozen as complete. Scoring preserves
original outputs. The rubric remains **0 irrelevant, 1 partially relevant,
2 directly relevant, UNKNOWN**; relevance is grade ≥1 and gains `2**grade - 1`.
Pooled nDCG@10 needs complete known labels and positive ideal DCG. General legacy
0–3 imports remain compatibility interfaces, not new primary annotations.

Relevance, independent factual/claim support, exact evidence delivery/retention/
recovery, and computational efficiency remain separate measurements. Blank
claim pools use the existing assessment interface. Exact matches do not prove
semantic support; unseen corpus text is not a memory failure. Reported/unreported
tokens, reasoning usage, latency, edit overhead, failures and gross costs remain
distinct from quality. No combined quality/efficiency score is introduced.

Remaining requirements are independent question/source review, reviewed public
factual tasks when measuring grounding, blinded output-inclusive relevance and
claim judgments, and explicit paid authorization. Offline tests establish no
retrieval, grounding or memory improvement.
