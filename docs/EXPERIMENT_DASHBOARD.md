# Comparative experiment dashboard

Implemented November 3, 2026. This is an interface to the existing controlled
benchmark, pressure fixtures, scorers and diagnostics. Diagnostics establish
software behavior; they do not establish retrieval or model superiority.

## Launch and navigate

From the project directory:

```bash
./launch-dashboard.sh
```

The launcher uses the project virtual environment, binds to localhost and disables
usage telemetry. If port 8501 is already occupied:

```bash
OPENALEX_DASHBOARD_PORT=8502 ./launch-dashboard.sh
```

Open <http://localhost:8502> for the instance started during implementation.
The equivalent direct command is:

```bash
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502 --server.headless true --browser.gatherUsageStats false
```

The default workspace is **Experiment Dashboard**. The sidebar also contains
**Individual research**, **Long-Horizon Testing** and **Saved Experiments**.
API credentials stay in the server configuration; the dashboard does not accept
or display keys. The configured hosted model remains Terra. Provider switching
and live OpenAlex acquisition are not part of this implementation.

## Supported experiments

| Selection | Existing execution backend | Scope |
| --- | --- | --- |
| Individual research | Existing research worker and OpenAlex interface | Interactive demonstration with explicit hosted-cost confirmation; reserved final questions are blocked |
| Comparison / Five-method comparison | `run_experiments`, controlled-five suite | Selected architectures, shared settings, isolated run directories |
| Graph ablation | Same comparison runner | Suggested conventional LLM and graph-enabled LLM pair |
| CLM ablation | Same comparison runner | Suggested conventional LLM and CLM pair |
| Long-horizon pressure | `context_pressure.run_pressure` | Offline controlled observation stream, measured pressure and recovery |
| Factual evaluation | Existing factual protocol and assessment scorers | Offline scorer tests in Diagnostics; approved public probes in empirical comparisons |
| G1–G5 / C1–C7 diagnostics | Existing graph fixtures and CLM/controlled-graph tests | Fixed, offline correctness checks |

The five architectures are Standard OpenAlex, conventional LLM, graph-enabled
LLM, CLM-inspired LLM, and graph + CLM LLM. Select all four agent architectures
for the intended 2×2 graph/CLM comparison. Subsets support conditional contrasts;
the equivalence panel warns that they cannot establish both main effects and the
interaction. Standard OpenAlex uses its existing single-search baseline.
The graph and context interventions retain their intended differences. All
controlled methods retain the existing final lexical ranker.

## Run an offline comparison

1. Select **Experiment Dashboard**, **Diagnostics**, and **Comparison**.
2. Keep the diagnostic question `retrieval therapy`. Select any supported
   methods, or all five. Start with one trial, seed `0`, and three actions.
3. Inspect **Configuration equivalence and maximum usage**. Budgets, model
   settings and the configuration/protocol digests are recorded before execution.
4. Click **Run Comparison** once. Status polling reads committed events and files;
   it does not simulate a percentage. Streamlit reruns reuse the existing job.
5. Inspect results, context trajectories, graph inputs, actions and raw manifests.
   Use **Prepare another experiment** to explicitly start a separate submission.

Diagnostics use the existing synthetic G1 corpus, regardless of a previously
selected empirical corpus. Its known fixture relevance grades apply only to its
exact question. Changing that question makes relevance **NOT EVALUATED**.
Hosted settings are not applied to the scripted model. Offline factual and G/C
test selections run fixed existing test fixtures with their own configurations;
the comparison budgets do not rewrite those tests.

Every output receives a fresh UUID under `data/experiment-dashboard/`. An output
name is a safe slug, not an arbitrary path. Inputs may be selected only from
approved project `data/`, `benchmarks/`, and `docs/prepilot/` locations; redirected
paths that escape those locations are rejected. Neither shell commands nor API
keys are UI input fields.

## Long-horizon testing

Select **Long-Horizon Testing**, then choose pressure, architectures, active
working-context capacity, observation length, initial evidence position,
distractor size, optional correction position, external archive availability,
trials and seeds. The full-input, per-arm cumulative-token, and global-token
limits also apply. Stream acquisition/call limits are derived from the horizon
and displayed; disabled benchmark acquisition controls do not affect this stream.

The fixture uses the same conventional and CLM controllers as before. It adds
configurable stimuli and observations of their behavior, without changing the
planner strategy to favor either architecture. Corrected evidence updates the
same canonical work and is evaluated against the new datum. A controller may
still retain a stale fact; that is recorded as an error.

Pressure is validated using actual named-study-tokenizer measurements. Both the
incrementally delivered observation volume and distinct delivered source text
must satisfy the configured ratio to capacity: low below 1, medium above 1, high
above 8. Required source versions must actually have been presented, the intended
horizon must complete, and common source archives must agree across arms. A
large paper count alone does not establish pressure. Some custom placements,
small distractors, corrections, or restrictive budgets legitimately produce an
**INVALID PRESSURE CONDITION**. Raw results remain available; invalid conditions
are excluded from aggregate charts.

The table distinguishes retained evidence, omitted/removed evidence, audit
storage, operational archive availability and successful recovery. Disabling
archive reads disables the agent's recovery operation while retaining raw
evidence for auditing. Exact recovery of the initial probe is separate from
correctness after a correction. Unattempted recovery is **NOT EVALUATED**.
Scripted study tokens are not presented as provider billing tokens or API cost.

## Inspect results and context

**Side-by-side results** offers question/trial filters, per-method rows,
descriptive charts and the existing paired comparison helper for complete
schedules. Variants and action budgets stay separate. It reports available
precision, nDCG, unique relevant papers and relevant papers per action; factual
answer accuracy, coverage, grounded completion and claim classifications;
actions, runtime, context overhead, usage, costs, failures and fallbacks.
Missing or unverified judgments yield **NOT EVALUATED**, never an inferred zero.
Reported provider usage and estimated unreported usage are separate. Unknown
pricing remains unknown. Primary factual accuracy remains distinct from
conditional coverage and claim metrics.

**Context evolution** compares actual saved requests across architectures. Select
a call to see its active working context, system/input/parsed output, recovery
payload and requested/committed/rejected/failed operations. Charts show recorded
request sizes, with named study-token counts where captured. Quote-presence
counts only test literal text from recorded evidence; they do not establish
semantic support or hidden model knowledge. Editable working context is separate
from authoritative OpenAlex metadata, graph records and raw observation archives.

**Actions and graph behavior** shows selected actions and arguments, discovered
work IDs, repeated action signatures, actual graph-derived planner inputs,
rankings and complete event traces. The graph affects planning in the current
controlled design; a graph-ranking effect is not claimed.

## Scientific readiness and approval

The existing development packet remains unreviewed: 117 relevance rows and three
factual-evidence rows require independent human completion. This implementation
does not complete those worksheets or approve any research experiment.

**Development pilot** and **Final research evaluation** fail closed unless the
existing review validator passes, imported relevance labels exactly match the
adjudicated worksheet, corpus and public questions/probes match the reviewed
bundle, and coordinator approval pins the exact configuration, source and
provider profile. Paid confirmation never substitutes for scientific approval.
Final evaluation additionally requires 10–20 held-out questions, at least three
trials and explicit holdout-independence attestation.

A coordinator approval JSON requires these fields:

```json
{
  "approved": false,
  "approver": "",
  "approved_at": "",
  "category": "Development pilot",
  "questions_sha256": "<exact question file hash>",
  "corpus_sha256": "<reviewed corpus fingerprint>",
  "spec_sha256": "<configuration equivalence panel>",
  "source_sha256": "<configuration equivalence panel>",
  "provider_profile_sha256": "<configuration equivalence panel>",
  "heldout_independence_attested": false
}
```

This is an unapproved template, not an approval. A responsible coordinator must
review the actual completed packet and protocol. Approval is invalidated by
changing pinned settings. Hosted runs additionally require explicit cost consent
and complete server settings. A dollar ceiling requires known positive provider
prices; a zero/unknown configured price is not evidence of a free hosted run.

Empirical artifacts retain copies of the actual review packet and approval.
Saved-result loading checks those copies, source hashes, corpus/question pins
and label equality again. A saved `ready: true` flag alone cannot enable quality
metrics. Old runs without this provenance remain inspectable as
**UNVERIFIED SAVED EXPERIMENT**, with quality withheld.

## Saved comparisons, assessment and exports

Select **Saved Experiments** to inspect multiple existing runs without execution.
The same result/context/trace views work for completed and available partial
artifacts. **Score new assessed copy** invokes the existing scorers in a fresh
directory, preserving its parent. Empirical relevance imports must still match
the independently adjudicated snapshot. Independent claim scoring additionally
requires a coordinator claim-review JSON with the exact claim-CSV SHA-256 and
the actual named assessors, for example this unverified template:

```json
{
  "claims_sha256": "<64-character SHA-256 of the reviewed claim CSV>",
  "review_status": "pending",
  "independent_human_review": false,
  "adjudicated": false,
  "approver": "",
  "reviewer_ids": []
}
```

Scoring requires `review_status: verified`, true review/adjudication flags, a
named approver and nonempty reviewer IDs matching the CSV. Do not use generated
model judgments to fill these fields. The reviewed inputs and provenance are
copied with the assessment. Assessment does not call a model or retrieve papers.

Export CSV summaries, PNG comparison figures, pressure records, manifests and
paired descriptive comparisons from their respective views. Summary CSVs and PNG
metadata include source experiment, configuration and evaluation status. Raw
paths remain traceable to the original recorded artifacts. Summary tables use
NOT EVALUATED for missing values and escape spreadsheet formula prefixes.

## CLI equivalence and reproducibility

The regular benchmark CLI and dashboard construct the same `BenchmarkSpec` and
call `run_experiments`; there is no second retrieval/evaluation implementation.
The regular CLI now accepts `--methods` and `--trial-seeds`. The complete guarded
dashboard adapter is also executable from a saved specification:

```bash
.venv/bin/python -m openalex_agent.experiment_dashboard --spec data/experiment-dashboard/EXPERIMENT/dashboard.json
```

This creates a fresh output. A hosted specification additionally requires
`--confirm-paid` and all the same review/budget gates. Recorded seeds identify
trials; no unsupported hosted sampling seed is transmitted and identical hosted
responses are not promised. Synthetic stimulus seeds do control their fixture.
Model settings, corpus, questions, run states, requests, outputs, errors and
source snapshots remain available for auditing.

## Execution limits and cancellation

Each architecture has isolated controller/context state and run artifacts.
Per-run existing budgets and dashboard-wide token/cost limits are enforced.
Hosted preflight reserves measured input plus the configured maximum output;
accounting uses provider usage when available and explicit estimates otherwise.
These estimates cannot guarantee an exact provider invoice.

**Cancel experiment** stops cooperatively at safe boundaries. An in-flight
hosted request may finish first. Pressure cancellation occurs between arms;
fixed diagnostic test subprocesses are terminated as owned process groups.
Cancellation, budget exhaustion and failures preserve committed state, requests,
events and completed metric rows. Polling reports actual recorded status.
Ordinary reruns do not submit another job. Interrupted jobs are inspectable but
there is no general resume or distributed queue implementation.

This is a local, single-process tool without multiuser authentication or
account-wide budget coordination. The individual research page retains its
existing execution behavior; experiment cancellation applies to dashboard jobs.
Human independence is attested by the review protocol, not proven by software.
Offline correctness checks do not validate hosted latency or live OpenAlex
availability. No paid calls or final empirical evaluation were run for this task.

## Implementation and verification

New files:

- `openalex_agent/experiment_dashboard.py`: guarded orchestration, safe artifacts,
  approval snapshots, read-only inspection, assessment copies and exports.
- `openalex_agent/dashboard_ui.py`: comparison, pressure, context and saved views.
- `openalex_agent/experiment_spec.py`: shared immutable CLI/UI benchmark spec.
- `openalex_agent/execution_control.py`: cooperative stop signal using existing
  controller checkpoint handling.
- `launch-dashboard.sh`: local virtual-environment launcher.
- `tests/test_experiment_dashboard.py`, `tests/test_dashboard_ui.py`: backend and
  real Streamlit AppTest coverage.
- This document and validation records under
  `data/dashboard-validation-2026-11-03/`.

Modified files:

- `openalex_agent/web_ui.py`: workspace navigation and explicit hosted consent.
- `openalex_agent/cli.py`: shared spec and method/trial-seed options.
- `openalex_agent/experiments.py`: seed records, scheduling cancellation and
  accurate owned-output failure statuses.
- `openalex_agent/context_planner.py`: optional diagnostic archive-read disabling;
  default operational behavior preserved.
- `openalex_agent/context_pressure.py`: configurable stimuli, actual token
  accounting, recovery observations and invalid-condition retention.
- `tests/test_web_ui.py`, `tests/test_experiment_repeats.py`: consent fixtures and
  accurate failed-manifest expectations.
- `README.md`: dashboard entry point.

Verification uses deterministic fixtures, mocked HTTP responses and actual
Streamlit AppTest rendering. It covers architecture selection, shared backend
specification, isolated state, multi-method execution, metric masking, pressure,
raw preservation, cancellation/failure, safe paths, rerun deduplication,
scientific gates, saved comparisons, assessment provenance and frozen protection.
Offline comparison, pressure, graph/context diagnostics, factual scorer checks
and CLI replay also produced recorded artifacts. The release suite command is:

```bash
.venv/bin/python -m pytest
```

Final result: **412 passed in 89.71 seconds**. The test log and preservation
verification are in `data/dashboard-validation-2026-11-03/`. The original
385-test suite is retained. All **18,788 historical files are unchanged**, with
zero missing files against the pre-implementation hash inventory. The frozen three-query
pilot manifest remains pinned to
`fdd43b90bafab8756f6ca3bc09041a21815d5c2f30ce883885f899d17e57bdc3`.
Retrieval, graph infrastructure, ranking and evaluation algorithms are unchanged.
