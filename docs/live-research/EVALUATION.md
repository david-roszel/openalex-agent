> Standalone live-research development design. Its standard/conventional/combined proposal is distinct from the primary controlled-five study. This document is not a claim that a full campaign completed or that approval/annotations are supplied. See [actual results and limitations](../RESULTS_AND_LIMITATIONS.md).

# Exploratory end-to-end evaluation protocol

This is an independent LiveDemo evaluation, not the V3 scientific campaign. Its purpose is to measure a small submitted agent/system honestly, without architecture superiority claims.

## Frozen inputs and scope

`benchmarks/live_submission_v1/questions.json` contains eight fixed questions in eight domains, four required components per question, and two predefined target publications per question. Selection used domain coverage and canonical literature comparisons before any new comparative execution. This is a convenience sample, not representative sampling or a held-out human relevance set. One initially unverified ecology DOI was corrected through bibliographic verification before preparation; the correction is logged in the manifest. No question or target was selected by measured agent performance.

Target identity is exact DOI, exact normalized title plus the predefined year, or an independently resolved canonical OpenAlex ID. There is no fuzzy matching and no retrospective target addition. OpenAlex ID resolution is separately recorded before hosted execution. Publication identity is not scientific support. Identity lookups are never supplied to research agents as oracle evidence.

The full plan has 24 trajectories (8 questions × A/B/C); 16 require hosted inference. The smaller predeclared option uses Q01–Q04: 12 trajectories, 8 hosted. There is one execution per question/variant and no numerical provider seed claim. The question is the comparison unit; calls, papers, claims and context transitions are not independent samples. Analyses are descriptive. No confidence intervals, significance tests or performance improvements are claimed from mocked tests or missing data.

Each preparation saves immutable `plan.json` with questions, resource/provider settings, implementation hashes and runtime package versions. A changed source/provider/dependency configuration requires a new preparation. There is no approval file after preparation. Approvals bind the plan hash and a separate numeric LiveDemo ceiling. Current full and smaller plans are exported as `plan.json` and `small-plan.json` in this directory.

## Configurations and resources

- **A / standard:** one pinned, nonagentic OpenAlex keyword query per question, up to 30 records, then the shared lexical ranking. No LLM planning, graph/CLM maintenance or answer generator. Answer metrics are N/A.
- **B / conventional:** the shared question decomposition, six-tool metadata acquisition, gap-aware continuation, ordinary selected context and final synthesis/revision. Graph memory and CLM editing are disabled.
- **C / combined:** the same question, acquisition eligibility, numeric ceilings and synthesis/revision policy, with directed citation graph observations and CLM-inspired file editing/recovery.

B/C use Thorough Research: at most 18 actions, six records/action, 144 unique admitted metadata records, 40 OpenAlex attempts, 28 total model calls, 150,000 total inference tokens, 24,000 serialized payload characters, 600 seconds and $1.50 per-run hosted bound. The 28 calls include analysis, all edits/recovery, one reserved final generation and one reserved content repair. There is no independent five-round stop. The source/observed metadata limits and shared tools remain the same between hosted variants. Graph and CLM can consume context/calls, so useful work need not be equal. Standard deliberately has a different nonagentic operating policy and no answer cost; do not describe it as a matched hosted factorial arm.

The source regime is live OpenAlex, with per-run metadata, timestamps and database snapshots retained. Future index changes can alter a new retrieval. Artifact reanalysis is reproducible; exact future live retrieval is not guaranteed. There is no hidden V3 cache or large frozen corpus dependency.

## Metrics and operational definitions

**Retrieval:** fixed-target Recall@10 in the final lexical top ten, target identities found, unique records returned/admitted, and structured source excerpts sent at the actual attempted HTTP boundary. A source ID in an action or graph edge alone does not count as a paper excerpt. Quote-traceable component coverage is a model assessment, not a human completeness score. Recall is unavailable when retrieval failed; failures remain in the operational completion denominator. Conditional means are explicitly labeled as available-run metrics.

**Answer:** presence of a final answer; structurally validated synthesis; unique traceable academic citations; untraceable published citations; limitations; all required question components queued for independent review. Every published claim in substantive answer sections must cite an encountered source and quote a supplied excerpt. Unsupported scientific claims and semantic completeness remain `null / UNVERIFIED` until independent review. A matching quote is not semantic verification. Failed or absent synthesis is recorded, not silently replaced with successful literature retrieval. Standard's answer metrics are N/A.

**Reliability:** all planned/attempted/completed/partial/failed/interrupted/not-run states, retrieval/planning failures, invalid/repeated action rejections, final failure and reached limits. “Premature stop” is only a declared proxy: planner STOP with unresolved model-assessed components. Resource exhaustion can yield a qualified partial answer and is distinguished from a crash. All failed cases are retained; unattempted cases are missing, not zeros.

**Efficiency:** actions, OpenAlex attempts, all-stage model calls, provider-reported input/output tokens, elapsed seconds, context overhead, and usage-based hosted cost upper bounds. Costs use reported tokens at conservative configured prices and are not invoices. OpenAlex monetary reports are separate; missing cost reports stay unknown. Failure latency is not successful research throughput.

**Mechanisms:** actual directed citation edges stored and verified edge pairs transmitted; CLM edits proposed/accepted/committed; accepted edits visible in later requests; observations archived; recoveries accepted and later transmitted. These metrics establish functioning/visibility, not causal reliance, research effectiveness, or superiority.

## Spending, execution and recovery

The ledger is a new independent LiveDemo ledger, with no V3 access or allowance. It reserves UTF-8 input-byte and maximum-completion cost bounds before transmission, then reconciles reported usage at conservative prices. Unknown/inconsistent usage and interrupted in-flight requests retain their holds and block further paid work. Duplicate request keys cannot replay. OpenAI direct calls pin Standard. There are no credit/discount assumptions and no automatic HTTP inference retries. The single content-repair opportunity is within the original research allowance and never retries an unknown-billing or transport failure.

A runner lock prevents duplicate campaign runners. Every trajectory writes RUNNING state before work and commits its result incrementally. Resume skips terminal cases. A leftover RUNNING case is marked INTERRUPTED and never re-executed automatically; remaining cases proceed only when spending is known. Mocked diagnostics and real evaluation must use separate campaign directories. All planned rows remain in `results.json`, including unattempted rows after an access or spending stop.

Hosted execution first requires a successful actual OpenAlex connectivity check and complete target identity verification. Thus failed retrieval access cannot burn an initial hosted planning call in the benchmark. The standalone research UI remains separately bounded and its failed attempts are preserved.

## Reproduce and review

See README for exact `prepare`, `validate-targets`, `authorize`, `run` and `report` commands. `report` makes no inference requests and regenerates JSON, CSV, Markdown and LaTeX artifacts from saved trajectories. Per-claim and per-component review queues remain UNVERIFIED until a real independent reviewer completes the rubric.

The current actual-attempt dataset has eight standard cases that failed at the network connection stage, no successful fresh OpenAlex retrieval, and sixteen hosted cases not run. The token/model pipeline diagnostics use authentic historical metadata with explicitly mocked inference, stored separately. Historical V1 is not substituted for the new evaluation. A paper must describe this state as incomplete exploratory evaluation, not a successful comparative study.
