# Five-method research evaluation protocol

The current development-only preparation is documented in
[evaluation status](EVALUATION.md) and
[PREPILOT_ANNOTATION_GUIDE.md](PREPILOT_ANNOTATION_GUIDE.md). Independent review is
pending. The opt-in `--factual-questions` public manifest gives every agent one
terminal response opportunity and reserves one of the total model calls.
Planning claims remain traces; fixed-denominator task metrics are the primary
factual outcomes. Existing commands without this option retain legacy claim
behavior and do not establish equal answer opportunities. Public probe files
must contain no verified answers or labels. These additions do not authorize a
hosted experiment or validate the proposed final-study questions.

Current correction/readiness record:
[version and limitations](RESULTS_AND_LIMITATIONS.md).
Do not start outcome-bearing pilots merely because engineering gates pass.

Corrected analysis retains variant/trial identity, validates completed schedules,
and restricts primary factorial contrasts to `full`. It exports graph and CLM
marginal main effects as well as conditional effects and interaction. Citation
corpus membership, claim-time observation/access and independently assessed
support/contradiction are separate; missing provenance remains unknown.
Optional token limits count the pinned study encoding, not undocumented
provider-internal framing. See the follow-up for measured diagnostics and gates.

**The research project is not complete.** The infrastructure can now compare
graph information and model-controlled context. Independent held-out relevance
and claim-support judgments, a representative corpus and the final experiments
remain outstanding. No current result establishes real CLM or graph superiority.

## Methods and controls

Use `benchmark --suite controlled-five` for the primary study:

| Method | Agent | Graph | Context management |
| --- | --- | --- | --- |
| `standard` | no | no | ordinary OpenAlex/corpus search |
| `llm` | conventional LLM | no | externally selected observations/history |
| `llm_graph` | conventional LLM | yes | same policy plus graph-derived information |
| `clm` | CLM-inspired | no | model-editable active-context file |
| `clm_graph` | CLM-inspired | yes | editable context and graph observations |

The four agent cells have the same model, inference settings, raw metadata,
six retrieval actions, metadata-based eligibility, original-question guards,
path-depth policy, action/inspection/HTTP/call limits and final lexical ranking.
Both context policies can read slices of previously delivered observations.
Graph-free cells do not construct a graph or compute graph scores. Record
actual token/cost consumption and failure behavior, rather than assuming
equal use of the ceilings. The standard baseline remains non-agentic and can
use its inspection allowance in one search; iterative agents use their page
size per acquisition. That method difference is explicit.

The original `standard/reformulated/iterative/graph` suite remains the default
and is still available with `--suite legacy`. Its search-only iterative method
is not the new graph-information control. The frozen pilot and earlier
experiments retain their original identities, settings and outputs.

## Development and held-out design

Two new files propose, but do not freeze, a study split:

- `benchmarks/research_development.json`: four development questions.
- `benchmarks/research_heldout.proposed.json`: sixteen proposed held-out questions.

They preserve existing question text and add unvalidated design annotations for
topic, ambiguity, subtopic count, citation complexity, temporal constraints,
evidence requirements and trajectory length. Review or replace questions before
using them as held-out data. A `proposed_heldout` label is not proof that a
question was unseen or never used in tuning. The final question file must
explicitly declare `split=heldout`, contain about 10–20 reviewed questions and
remain separate from development. `report --final` rejects undeclared or mixed
development questions in the controlled suite.

Acquire a real corpus spanning all selected areas, targeting approximately
10,000 unique works. Freeze the papers, canonical IDs, acquisition queries,
filters, timestamps, count and checksum. Document missing targets, query bias
and available full-text/abstract evidence. The old 39-paper pilot and synthetic
capacity fixtures are not substitutes for a representative final dataset.

Pre-register the model/version, context protocol, scoring weights, binary
relevance threshold, output K, budgets, evidence task, exclusions and primary
contrasts. Tune only on development data. Do not change settings after seeing
held-out results. A null result or a regression must remain reportable.

## Repeated controlled experiments

Terra remains the default hosted model. Keep its profile fixed; use Qwen or
Gemini only as separately named follow-ups if justified. Use the same frozen
corpus and settings for all four agent cells, budgets 5/10/20, and **at least
three trials** per question/method/budget. The runner defaults to three and
rotates method order. Sixteen questions imply 720 runs, of which 576 are agent
runs; twenty imply 900. A 20-call ceiling can still permit many hosted calls.
Estimate cost before the final run; the commands below are preparation, not
authorization to execute the unready study automatically.

First validate a small development subset and low budgets. Do not use final
independent judgments to guide retrieval. For the eventual final corpus and
reviewed held-out file:

```bash
.venv/bin/python main.py --db data/final-research.sqlite benchmark \
  --suite controlled-five --planner hosted \
  --corpus data/corpus/final --questions data/study/heldout-frozen.json \
  --action-budgets 5 10 20 --trials 3 --max-rounds 20 \
  --max-inspected 200 --page-size 20 --top-k 10 --max-api-requests 50 \
  --max-llm-calls 20 --context-chars 16000 --max-total-tokens 100000 \
  --active-context-tokens 4096 --input-tokens 8192 \
  --require-evidence --output data/experiments/final-five-terra-v1
```

`--require-evidence` asks for a structured evidence record or explicit
insufficiency/abstention at each agent decision. Fix that requirement across all
four agent cells. The standard baseline emits no model claims; its factual
rates remain unavailable rather than being scored as perfect accuracy. If
using optional claims instead, report elicitation policy, opportunity count
and claim coverage; silence is not evidence of greater factual reliability.

`--max-llm-cost-usd` optionally adds observed dollar stopping, but requires
positive configured token prices. It is not a prepaid hard cap. Token preflight
is estimated and can differ from provider usage; actual reported and unknown
usage remain separate. Context editing/recovery uses the shared call budget,
not additional hidden calls. Keep failure/fallback rows in the primary system
comparison; label any secondary valid-pair analysis and disclose exclusions.

All controlled comparisons use the fixed corpus, not live OpenAlex. Run live
demonstrations separately and report actual HTTP/cache counts. Every new
experiment needs a fresh output directory. The manifest includes source copies
and hashes, model configuration, limits, questions, corpus and run schedule.
Each run retains ranked IDs, acquisition history, requests, outputs, events,
context snapshots, token/cost diagnostics, failures and fallbacks.

## Independent relevance and subtopic assessment

Pool the union across all methods, trials and budgets, deduplicate and hide
method identity. The pool includes inspected papers as well as ranked outputs
so efficiency/drift can be judged. The primary pooled rubric is 0 irrelevant,
1 partially relevant, 2 directly relevant, with UNKNOWN for unjudged evidence.
Binary relevance is grade ≥1 and graded gains are `2**grade - 1`; see
[the pooled protocol](POOLED_RELEVANCE_PROTOCOL.md). Legacy 0–3 imports remain
available for historical compatibility, not new primary-study annotations. Preserve
original sheets, obtain a second independent assessment on a subset, record
agreement/disagreements, then adjudicate conflicting grades before import.

```bash
.venv/bin/python main.py pool data/experiments/final-five-terra-v1 \
  --output data/assessment/relevance-blind.csv
.venv/bin/python main.py judgments-import data/assessment/relevance-adjudicated.csv \
  --scope pooled --output data/assessment/relevance.json
```

Unjudged papers remain unknown. Report coverage and precision bounds. Recall
is relative to the judged candidate pool unless every corpus paper was judged
for every question. Do not call pooled recall exhaustive scholarly recall.

Subtopic coverage uses separate independent labels, not the OpenAlex topics
shown to the agent. Example format:

```json
{"source":"curated","queries":{"q01":{"expected":["mechanism","evaluation"],
 "papers":{"https://openalex.org/W123":["mechanism"]}}}}
```

These are format examples, not real labels. Use real IDs from the frozen corpus
and independently defined subtopics. The agent never receives this file.

## Independent factual and citation-support assessment

```bash
.venv/bin/python main.py claim-pool data/experiments/final-five-terra-v1 \
  --output data/assessment/claims-blind.csv
```

The blinded pool includes the statement, claimed work ID, model-provided quote,
insufficiency flag and available authoritative title/abstract. Method identity
is hidden. Judges must verify support against the actual source, not assume a
model quote is accurate. Valid identifiers and matching quotations are
structural diagnostics, not factual proof. Abstracts may be insufficient for
claims about detailed experimental results; record that limitation or examine
the source under a documented assessment protocol.

Fill `judgment` with `supported`, `unsupported`, `contradicted` or `abstain` and
record `assessor`. Optional independently judged fields
`factually_correct`, `citation_support_correct` and `appropriate` use
`true/false`; blank means unknown. Unsupported need not mean factually false.
Record appropriate and inappropriate abstentions separately. No claims yields
undefined accuracy/support rates, not 100%. Retain judge originals and
adjudicate conflicts. Provenance fields cannot certify actual independence.

```bash
.venv/bin/python main.py score-experiment data/experiments/final-five-terra-v1 \
  --labels data/assessment/relevance.json
.venv/bin/python main.py score-research data/experiments/final-five-terra-v1 \
  --labels data/assessment/relevance.json \
  --subtopics data/assessment/subtopics.json \
  --claim-judgments data/assessment/claims-adjudicated.csv
.venv/bin/python main.py report data/experiments/final-five-terra-v1 --final
```

Scoring is offline and never sends judgments to the agent. It rejects
out-of-corpus labels, conflicting claim grades and incomplete claim assessment
rather than silently treating unjudged claims as false. Source assessment
provenance is recorded separately from retrieval artifacts.

## Metrics and statistical interpretation

Report distinct groups, without an opaque combined quality score:

- Retrieval: P@10, graded nDCG@10, judged-pool recall/coverage, independently
  labeled subtopics, unique relevant discoveries and actions per relevant paper.
- Factual reliability: independently judged truth accuracy, supported,
  unsupported and contradicted claim rates, citation-ID validity, citation-to-claim
  support accuracy and conditional appropriate-abstention rate. Include
  denominator/assessment coverage and answer opportunity counts.
- Context: verbatim retention and exact recovery on diagnostic probes, active
  size, edits/rejections, evidence lost through editing, archive reads and
  context-management model/local overhead. Do not infer semantic retention
  from a valid ID or a shorter context alone.
- Efficiency: actual and estimated/unreported tokens, known/unknown costs,
  mean/median response latency, total time, HTTP requests, useful papers/action
  and supported claims per known unit cost.

The runner exports per-question/trial results and repeatability. The research
scorer exports `research-metrics.csv` and `research-paired-comparisons.json`.
Average trials **within each question first**; repeated calls are not independent
research questions. Primary paired contrasts are graph–conventional,
CLM–conventional, graph+CLM–graph, and graph+CLM–CLM. The interaction is
`(clm_graph - llm_graph) - (clm - llm)`. Paired question-bootstrap intervals are
reproducible; fewer than two paired questions yields no interval. A higher
mean is not a significance test. Pre-register confirmatory contrasts, distinguish
exploratory comparisons, and account for multiple comparisons where appropriate.

The final empirical milestone requires this reviewed dataset, independent
judgments, repeated experiments, paired analysis and a report covering failures,
null effects and regressions. Engineering fixtures and the hosted synchronization
smoke fulfill none of those quality-study requirements by themselves.
