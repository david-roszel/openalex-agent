# OpenAlex Controlled Scholarly Evidence-Retention Benchmark — V2

**Status:** Complete pre-execution experimental design; **no empirical V2 observations collected here**.  
**Protocol identifier:** `openalex-memory-pressure-terra-v2-design-1`  
**Intended model:** previously validated GPT-6.1 Terra provider, configured and version-pinned locally by Codex.  
**Financial authorization:** **$68 maximum new gross Terra exposure across the entire Terra V2 program**, not $68 for this component alone. No inference authorized by this document.  
**Scientific scope:** Objective *development* mechanism study. Not an independent human-annotated held-out relevance or hallucination benchmark. Not a test of Meta's official Context Language Models implementation.

## 1. Rationale and separation from the natural-retrieval benchmark

The earlier OpenAlex objective-development campaign ran five architectures under maximum 3- and 10-action horizons. The long-horizon agents often stopped early, leaving limited context pressure. The graph methods delivered summaries and graph-ranked candidates but no **explicit structured citation edges** in their transmitted prompts (585 graph-method requests inspected, 0 explicit-edge deliveries). The prior report already envisioned eight fixed batches of three real papers, but recorded zero completed controlled memory probes. These observations motivate **a new, prespecified, mechanism-focused experiment**, not retroactive alteration of existing scores.

Two complementary studies must remain separately reported:

- **Natural scholarly retrieval:** 30 frozen tasks, five methods, action budgets 3/10, target discovery/citation retrieval and cost; agent chooses when to STOP.
- **This controlled memory study:** a fixed stream of genuine evidence, followed by delayed source-identity, year, and directed-citation questions. The scheduled stream continues regardless of the model's natural STOP preference. This is a controlled exposure study, **not** a natural agent-search trajectory. No live OpenAlex or Gemini requests are needed to assemble the fixed stream if the frozen corpus contains the records.

Do not pool the two studies' trajectories, averages or p-values. The Terra V2 study is a corrected follow-up informed by V1 observations; do not call it independent held-out validation.

### Research questions

- **M-RQ1 (retention):** With the same scheduled real-paper exposure, does graph memory, CLM-inspired context editing, or their combination improve *delayed answer accuracy without any recovery tool*?
- **M-RQ2 (recovery):** With the same original exposure and equivalent bounded archive access, which architectures recover previously encountered evidence most accurately?
- **M-RQ3 (mechanism):** Are verified edges and source facts available, included in the actual requests, retained/edited, and later recovered? How much do these operations cost?

### Prespecified hypotheses (directional; no assumption of confirmation)

- Graph availability may favor citation-relationship questions over isolated source facts.
- CLM editing may help retention under working-context pressure, but could also remove needed information.
- An interaction is plausible if CLM preserves graph-derived relationships, but can be negative or absent.

## 2. Design and statistical units

| Property | Frozen design decision |
|---|---|
| Source | Frozen real OpenAlex scholarly corpus and existing canonical-ID/citation records |
| Independent cases | **12**, exactly **two per each of the six domains** (if audit passes) |
| Initial repetitions | **1 per case × architecture**, to maximize independent-case count at fixed budget |
| Architectures | `llm`, `llm_graph`, `clm`, `clm_graph` (four hosted Terra methods) |
| Initial main matrix | **12 cases × 4 methods × 1 repetition = 48 trajectories** |
| Evidence per case | **24 distinct papers**, assigned to **8 ordered batches of 3** |
| Delayed questions | **6 fixed questions per case**, answered after batch 8 |
| Evaluation branches | **A: no recovery**; **B: recovery allowed**, forked from an identical frozen post-exposure state |
| Optional precision expansion | A **second full repetition across all 12 cases and all four methods = +48 trajectories**, never selectively chosen by outcome |
| Pilot | **2 distinct non-main cases × 4 methods = 8 trajectories**, separately labeled, charged to same $68 limit |
| Provider | Same pinned Terra model/inference configuration across four methods |
| Exposure sequence | Identical batches and ordering for all methods within a case |

**Important trade-off:** This deliberately revises an earlier six-cases × two-repetitions proposal. Twelve distinct cases improve task coverage at the same 48-trajectory total. One repetition leaves hosted-model randomness less characterized; optional full-matrix repetition can address that if time and funds permit. Twelve cases, not 48 trajectories or 72 question answers, are the independent observational units. The six domains are fixed and cannot support broad discipline-generalization claims.

## 3. Four-arm factorial controls

| Graph mechanism | No CLM editing | CLM-inspired editing |
|---|---|---|
| Absent | `llm` (G=0,C=0) | `clm` (G=0,C=1) |
| Present | `llm_graph` (G=1,C=0) | `clm_graph` (G=1,C=1) |

All arms share the same actual provider, prompts apart from mechanism-specific capabilities, evidence-feed format, context ceiling, action/tool-call budget, read-only document/archive recovery interface, final question set, parser, scorer and initial state. No arm gets undisclosed oracle facts or extra original papers.

**Graph access:** Every arm is exposed to identical source-record metadata and citation-reference facts legitimately present in the source stream. Graph arms alone materialize those **already exposed** reference facts into an indexed persistent citation graph and serialize eligible directed edges and provenance into working context. The two graph arms must have equivalent underlying graph records and graph-query capability. This tests the operational value of **graph representation, selection and availability**, not the value of gifting graph arms extra hidden knowledge. Preserve and log context displacement caused by graph serialization.

**CLM access:** Both CLM arms can perform the same validated model-initiated context edits (retain/compact/reorganize/archive/recover as genuinely supported by the V2 implementation), within identical active/full context limits. Non-CLM arms use the same prespecified deterministic context packing/compaction, without model-controlled editing. All arms have equal bounded read-only archive/document recovery in branch B. Graph-only and Graph+CLM may additionally query the same eligible graph, subject to the same **total** recovery-operation ceiling. This is an operational architecture comparison, not a perfectly isolated pure-memory causal experiment; archive indexing, tool affordances and text allocation must be reported.

**Model/sequence comparability:** Reuse one deterministic order of the four arms within each case, rotated across cases to mitigate provider-time bias. Use a unique fresh context state per case-arm. Hosted randomness is not claimed to be seed-controlled unless the actual Terra provider supports a verified deterministic seed.

## 4. Constructing the twelve real-paper cases (offline, then freeze)

Use the **frozen ~30,000-paper OpenAlex corpus**, not fresh live search. Reuse the six existing discipline labels from the original benchmark. Produce two independent cases per discipline using canonical OpenAlex records and verified citation edges. Proposed within-discipline strata: (a) relatively dense local citation neighborhood and (b) relatively sparse but still reachable chain. The design is **not** stratified based on which architecture previously performed well.

For each case:

1. Select three real, distinct works `A`, `B`, `C` such that **B cites A** and **C cites B** in the frozen citation metadata; verify direction from source records. Arrange first presentation: `A` batch 1, `B` batch 4, `C` batch 6. Ensure at least one additional verified directed edge between distinct exposed works for a separate one-hop question. If possible, have the second member of each domain pair differ in degree density or lexical overlap.
2. Select **21 additional unique works** from the same scholarly domain with similar record-completeness constraints. Include lexical/semantic near-distractors and ordinary domain papers. Do not use a model to choose distractors in response to benchmark outcomes. Publish the deterministic selection rule and seed before any scored Terra call.
3. Place the 24 works into eight 3-record batches. The three chain works are separated by intervening documents; final batches 7–8 contain distractors sufficient to displace early evidence under finite context. No repeated full-record re-exposure after its scheduled batch.
4. Freeze source IDs, years, authors if included, exact excerpt spans, reference direction, batch positions, derived questions/keys and all selection decisions. Hash each case and the entire manifest.
5. Exclude exact works serving as hidden target/anchor records in the original natural benchmark or its six pilot tasks if locally feasible; record any unavoidable overlap. Cross-case independence means distinct anchors, distinct key directed edges, and preferably disjoint 24-paper exposure sets. If disjointness is impossible within a domain, log overlaps and avoid overstating independence.
6. Reject cases with missing or conflicting year/ID fields, ambiguous multi-answer probe targets, non-verifiable edge direction, or duplicate paper IDs. Replace only under **prepublished deterministic eligibility rules before outcomes are observed**. Never replace a difficult case based on poor model results.

### Record format, identical across four arms

Each paper exposure contains: canonical OpenAlex ID; title; publication year; a **verbatim, provenance-tagged** abstract passage chosen by a deterministic corpus rule; and available verified references restricted to **previously exposed or simultaneously newly exposed works**. Citation facts are stated in ordinary source-record form for every arm. Graph arms can additionally organize these same facts into explicit edges. No information about future batches, hidden answer keys, or undiscovered papers may be leaked through graph expansion. For a citation between two exposed works, expose its canonical direction as soon as both ends have appeared.

Suggested excerpt target: approximately 220–300 tokens per work **after testing Terra's actual token/character ruler and the existing context limits**. Do not paraphrase, synthesize abstracts, or fabricate passages. If records are too short, select other eligible real records or declare reduced pressure; never pad with fictional scholarly facts. Limit the batch size so its three records plus required instructions always fit the per-call input cap.

### Pressure calibration (offline, outcome-independent)

Measure the serialized token and character lengths **before any scored model invocation**. Freeze the exact V2 active/full context bounds (the old Gemini profile used 4,096/8,192 tokens and 16,000 characters; verify how the Terra adapter implements these rather than assuming identical tokenization). Require aggregate evidence-stream size to exceed the active-context allowance sufficiently that keeping every passage verbatim is impossible by batch 8. Target ≥1.5× the active allowance under the binding dimension, with a measured check that each individual batch fits. If no case satisfies the planned pressure, report that before launching and adjust excerpts using one predeclared eligibility rule or reframe as low-pressure; do not label an unpressured test as a long-horizon memory challenge.

Keep per-step model input within the same defined bounds for all four arms. The evidence feed and selected context are **not** assumed identical: the agent's selected/retained context is an outcome. Audit whether all 24 scheduled records were actually transmitted at least once to each arm. A paper rejected by capacity admission does **not** count as successfully exposed.

## 5. Execution timeline for each case/arm

**Before batch 1:** Reset state. Announce only the generic goal: track source identities, years and citation relationships because future questions may concern earlier evidence. Do **not** reveal the six specific questions, target IDs or designated chain.

**For batch 1 through 8:**

- External harness supplies the same three source records, in frozen order, to every arm. The harness is responsible for delivery; no model-chosen searching is used in this controlled phase.
- After each delivery, provide a bounded maintenance/decision opportunity under the same model-call ceiling. Model-controlled edits are available only to CLM arms; a no-op is allowed. Do not force editing. Agent STOP or unwillingness to inspect is logged, but it does **not** terminate the external evidence stream. No model can inspect unscheduled corpus documents.
- For graph arms, add only citation edges whose endpoints are legitimately exposed; record edge candidates, selected/delivered edges, graph token allocation, and displaced evidence. The actual serialized request (or a safe hash+structural parser of it) determines whether an edge was visible.
- Every arm's persistent archive can store all previously exposed source records in a read-only durable evidence store; working context is finite. Ordinary archive recovery is disabled during the no-recovery endpoint. Context edits cannot modify original source records.
- Preserve a per-batch durable state snapshot and per-arm trace of actual evidence exposure, admission, context packing, edits, recovery calls, archived content and eventual STOP flags.

**After batch 8:** Create a single immutable *post-exposure snapshot* for this arm. Fork **two independent continuation branches** from that **same state**:

- **Branch A, no recovery (PRIMARY):** ask the six delayed questions in one strict JSON response. Disable all archive, graph-query, retrieval and additional context-edit actions during answering. The already-selected working context and the automatically included graph summary, if present at that frozen state, remain available. No other evidence can be fetched.
- **Branch B, bounded recovery (SECONDARY):** ask the identical questions using the same frozen pre-question state (not Branch A's edited/answered state). Allow up to **two total evidence-recovery operations** from the 24 exposed records, counting archive search, document reopen and any graph query against the same cap. Permit normal mechanism-specific editing if supported, but forbid outside corpus search, future evidence and oracle access. Answer in the same JSON schema.

Order the two branch calls using a prespecified balanced assignment (e.g., alternate A/B by case), but always fork from the same snapshot so answers do not leak between branches. The two branches are correlated outcomes of one case-arm, **not** independent trajectories. If snapshot cloning is not supported, implement it deterministically before paid execution; do not simulate branching by simply continuing from the first answer.

At most eight scheduled maintenance calls and two final-answer calls per arm plus two bounded recovery-operation interactions are planned; the implementation must define a strict **provider-call ceiling** (suggest 12 generation calls per case-arm) and a strict **total input/output-token ceiling** in the frozen protocol. A tool operation may require more than one provider round-trip; count every paid generation toward the common ceiling. If the endpoint cannot fit this ceiling, pilot-validate a revised fixed bound *before* main cases.

## 6. Six delayed questions and strictly objective answers

Each case has the same six *question roles*. The language is populated offline from actual frozen source passages; never fabricate metadata. Avoid copying answer IDs/titles into questions unless that is expressly part of the intended citation-relation cue. Assert that each answer is unique among all 24 exposed records.

| Key | Type | Source timing | Answer format | Scoring |
|---|---|---|---|---|
| **Q1** | Earlier source identification from a distinctive excerpt clue | Batch 1 | canonical `work_id` | Exact ID match |
| **Q2** | Publication year for a distinct previously encountered paper, identified by descriptive cue | Batch 2 | integer `year` | Exact stored year |
| **Q3** | Midstream source identification | Batch 4 | canonical `work_id` | Exact ID match |
| **Q4** | Directed one-hop citation relation between exposed sources | Batches 1–4 or another preregistered pair | `{citing_work_id, cited_work_id}` | Both IDs and orientation exactly correct |
| **Q5** | Two-hop citation chain spanning early, middle and later evidence | A (B1), B (B4), C (B6) | ordered `path` as `[C,B,A]` | All three IDs and direction correct; partial-credit components diagnostic only |
| **Q6** | Evidence calibration: whether an additional purported relation is **established by presented records** | After all batches | `SUPPORTED` or `NOT_ESTABLISHED` | Exact evidence-status match; six cases each status, prespecified |

**Q6 is about supplied evidence, not about whether a citation truly does not exist anywhere.** For `NOT_ESTABLISHED`, the correct response is not a categorical claim of absence. For six `SUPPORTED` cases use a distinct second verified edge; for six `NOT_ESTABLISHED` cases use a pair whose relation is **not established by the exposed records**, with offline checks that it does not appear in any exposed reference list. Counterbalance status across domain and citation-density strata as possible.

Questions must be generated from the *frozen evidence*, audited for uniqueness and comprehensibility **before model results**, and stored in a withheld answer-key file not accessible to the model. Only the question text, without correct answers, is sent at the end.

### Final response contract

Strict JSON object with exactly `q1`, `q2`, `q3`, `q4`, `q5`, `q6`:

```json
{
  "q1": {"work_id": "W..."},
  "q2": {"year": 2024},
  "q3": {"work_id": "W..."},
  "q4": {"citing_work_id": "W...", "cited_work_id": "W..."},
  "q5": {"path": ["W...", "W...", "W..."]},
  "q6": {"status": "SUPPORTED"}
}
```

The example values are **format placeholders, not real scholarly claims**. The benchmark scorer must use exact canonical-ID/year/direction checks. Parsing failure, wrong orientation, unsupported invented IDs or omitted fields score zero for the affected item; include the raw response and error category. No LLM judge. Answers cannot be considered proof of scientific semantic support or reduction in hallucinations.

## 7. Outcomes and scoring (freeze before execution)

### Single preregistered primary endpoint

**Delayed evidence accuracy without recovery (`Y_A`).** For each case-arm, compute the arithmetic mean of binary Q1–Q5 correctness from **Branch A**:

`Y_A = (correct(Q1)+correct(Q2)+correct(Q3)+correct(Q4)+correct(Q5))/5`.

Q6 is intentionally excluded from the primary endpoint so calibration does not dilute source/edge memory performance. The primary is a **retention-and-use** score: Q4/Q5 involve both remembered evidence and reasoning; it is not a pure measure of internal memory capacity.

### Prespecified secondary endpoints

1. **Recovery-enabled accuracy (`Y_B`)**: same Q1–Q5 mean on Branch B.
2. **Recovery uplift**: per-case `Y_B - Y_A`, with branch states forked identically.
3. **Non-relational memory**: Q1–Q3 mean (identity/year).
4. **Graph-relational evidence use**: Q4–Q5 mean (direct edge/two-hop path).
5. **Evidence calibration**: Q6 accuracy, plus error split for falsely declaring `SUPPORTED` versus unwarranted `NOT_ESTABLISHED`.
6. **Availability at answer time**: for each gold fact and directed edge, report `exposed_once`, `archived`, `selected`, `in_actual_final_input`, `recovered`, and `omitted_by_capacity` (do not infer model reliance solely from visibility).
7. **Memory trace**: model-proposed/accepted/committed edits, graph relations preserved or removed, archive/document/graph tool queries, duplicate/redundant retrievals.
8. **Efficiency and reliability**: gross USD, input/output tokens (explicitly include reasoning where reported), generation calls, median and tail response latency, wall time, failures, fallbacks and capacity rejections.
9. **Age/pressure diagnostics**: accuracy for probes whose support first appeared in B1/B2/B4/B6; plot actual context-availability and discarded/recovered evidence by batch. Avoid calling these repeated observations independent units.

The original paper's known-target Recall@10 and citation-chain discovery are **not** scores for this controlled stream: those belong to natural retrieval only.

### Operational failure and missingness

- A **terminal experimental failure** (provider error, invalid final output, exhausted common call budget) is retained as a failed trajectory. Under the predefined intention-to-treat (ITT) operational scoring, failed final answers score zero for Q1–Q5; also report correct preliminary evidence exposure and partial outputs separately. Report failure-rate disparities.
- A **never-attempted cell** due to budget/deadline is **missing**, never a fabricated zero. A completed case-arm with one missing item scores that item zero; do not equate unavailable provider telemetry with zero cost.
- Include complete-case paired analyses and sensitivity results alongside the prespecified ITT operational scores. Do not silently remove failures or cherry-pick matched blocks.
- A failure to deliver the scheduled 24 papers is a protocol violation/exposure failure and must be reported, even if the final answer happened to be correct.

## 8. Prespecified statistical analysis

For each independent case `t`, obtain the four primary scores:

- `Y00(t)` = LLM
- `Y10(t)` = LLM+Graph
- `Y01(t)` = LLM+CLM
- `Y11(t)` = LLM+Graph+CLM

Define paired factorial contrasts per case:

- `G(t) = ( [Y10(t)-Y00(t)] + [Y11(t)-Y01(t)] ) / 2`
- `C(t) = ( [Y01(t)-Y00(t)] + [Y11(t)-Y10(t)] ) / 2`
- `I(t) = Y11(t) - Y10(t) - Y01(t) + Y00(t)`

Report means over the 12 independent cases, individual case effects and paired effect distributions. Report the direct comparisons `Y10−Y00`, `Y01−Y00`, `Y11−Y10`, `Y11−Y01` as secondary.

Uncertainty: use a prespecified paired **task/case-level** bootstrap (10,000 replicates; fixed seed) preserving all four arms together. Where possible stratify within the six fixed domains (two cases each); acknowledge fragile intervals with only two cases/domain. Show percentile 95% exploratory intervals and exact/Monte Carlo paired sign-flip sensitivity where its symmetry assumptions are defensible; avoid definitive significance claims at N=12. For the three factorial primary-family contrasts, if p-values are displayed apply Holm correction; label any tests underpowered/exploratory. Since there is a single primary outcome (`Y_A`), secondary metrics remain exploratory.

If some of the 12 cases lack full four-arm results, report (a) operational partial status and (b) paired results on **complete four-arm case blocks** separately, never filling missing cases with zeros. If a second repetition is authorized prospectively, aggregate repetitions **within each case and arm first**; still N=12.

Prespecify reported contrasts and interpretation before looking at outcomes; no post-hoc threshold tuning, favorable-case exclusion or selective expansion. An interval containing zero implies inconclusive evidence, **not** equivalence. A positive effect under these automatically constructed cases is not proof of general scholarly retrieval or hallucination reduction.

## 9. Test and freeze gates before paid execution

All gates below must pass offline or with mocked provider responses:

1. Real corpus and citation edge provenance hashes are verified; no fabricated IDs or edges.
2. Exactly 12 main cases in six domains, with eight 3-paper batches, 24 distinct IDs/case. No main/pilot overlap in case IDs; known work overlap fully audited.
3. Every question has exactly one verified answer (or Q6 evidence-status truth), with no unintended ambiguity/alternate citation-chain answer.
4. The complete `case_manifest.json` and separate private `answer_keys.json` are hashed and immutable **before any scored Terra calls**.
5. Each paper batch fits actual context admission; aggregate stream exceeds available active context under measured bound, not just nominal word count.
6. The test harness delivers each paper to the actual model-input boundary at least once across the eight batches; missing delivery is flagged.
7. Both graph arms have the same eligible and indexed edges; positive fixture shows explicit edges present in final serialized model input (not only internal state); neither non-graph arm receives graph-only serialization.
8. Both CLM arms can actually commit a valid edit and show its persistence in subsequent true input (corrected V2 canary). Neither non-CLM arm can use the edit action.
9. Snapshot fork tests prove Branch A and B start from identical frozen state and cannot see one another's answers; Branch A has **zero** external recovery access; Branch B is capped at two operations against previously exposed evidence only.
10. Scorer passes positive, negative, malformed, reversed-edge and unknown-ID fixtures without an LLM judge.
11. Terra adapter's model ID, endpoint, inference params, actual token accounting and provider timeouts are pinned; test without sending requests. Any paid pilot requires separate phase authorization.
12. One campaign-wide **$68 gross exposure ledger** exists across Terra natural-retrieval pilots, natural runs, controlled-memory pilot and controlled-memory main runs. No phase-specific reset; conservative holds for ambiguous usage; no automatic retry of uncertain requests.
13. Output files use atomic writes and immutable source snapshots. Dashboard is read-only. Frozen V1 Gemini results are never modified.
14. The experiment, its hypothesis, and final question templates are frozen *before* Terra V2 outcome inspection. Pilot uses disjoint cases and may check **mechanical validity, latency and cost**, not tune tasks to improve metric results.

## 10. Execution and budget policy

Suggested scientifically useful order within one $68 Terra cap:

1. Finish graph-visible Terra engineering and offline source/probe canaries; **no paid requests**.
2. Run a **disjoint 8-trajectory memory pilot** (2 pilot cases × 4 hosted methods) after provider/accounting authorization; validate real pressure, exact edge delivery and state forking. Do not count pilot outcomes in main analysis.
3. Protect a complete 30-task × 5-architecture × 2-repetition **3-action natural retrieval stage (300 trajectories)**, if that is the approved main design.
4. Run this **48-trajectory memory main matrix** in twelve matched four-method blocks across all domains.
5. Run the planned **300 10-action natural retrieval cells**, subject to actual budget/time and the existing protocol.
6. Consider a second **full 48-trajectory memory repetition** only if authorized, budget allows, and the decision to expand is made without inspecting whether memory effects are favorable.

This scheduling is a suggested allocation, not permission to silently change any already frozen Terra V2 protocol. If a previously authorized experiment fixes a different order, document and reconcile that before launch. Include the memory pilot and all natural-retrieval requests in the **same** $68 ceiling. Make no assumptions about Terra's current prices, call counts or credit balance; use actual configured provider pricing and measured pilot cost. Maintain a conservative reservation for each request and stop before a new request that could breach the limit. Do not substitute an optimistic latency/cost estimate for a feasibility check.

Aim for **four complete arms on each case** rather than many unmatched observations. Schedule paired blocks round-robin across domains. Keep any dashboard/UI process strictly read-only.

## 11. Artifacts and provenance

Create a new path that cannot overwrite the Gemini campaign or natural Terra results, e.g.:

```
data/terra-v2-controlled-memory/
  PROTOCOL.md
  source_manifest.json          # frozen corpus hashes, version, sampling algorithm
  case_manifest.json            # 12 public-to-run cases; source texts in secure local snapshots
  answer_keys.private.json      # hidden from agent prompts, available to deterministic scorer only
  pilot_manifest.json           # disjoint two-case feasibility pilot
  cells/                       # atomic case × method × repetition outputs
  traces/                      # exposure/edit/graph/request-boundary provenance
  ledger/                      # links to ONE global Terra exposure ledger, not a second budget
  RESULTS.csv
  COMPARISONS.csv
  FACTORIAL_EFFECTS.csv
  EXPOSURE_AUDIT.csv
  PAPER_TABLES.tex
  FIGURES/
  ANALYSIS.md
  LIMITATIONS.md
  HASHES.sha256
```

Minimum required row fields in `RESULTS.csv`: `protocol_version,case_id,domain,graph_condition,clm_condition,method,repetition,branch,terminal_state,exposures_scheduled,exposures_actual,pressure_verified,graph_edges_eligible,graph_edges_delivered,edits_committed,recoveries_used,q1_correct,q2_correct,q3_correct,q4_correct,q5_correct,q6_correct,primary_score,known_input_tokens,known_output_tokens,gross_usd,cost_complete,provider_model,source_snapshot_sha256,cell_result_sha256`. Store raw responses securely, never include hidden gold fields inside model input.

Freeze the exact trial protocol and hash it. Record all changes as versioned amendments made **before** accessing scored outcomes. Supply an audit-ready run ledger including failed and skipped cells.

## 12. Acceptance criteria and what the paper can claim

A *completed benchmark design/implementation* must demonstrate: real-frozen evidence and answer keys, balanced 12×4 matrix, actual finite-context pressure, real graph-edge delivery, validated CLM edit persistence, independent endpoint branching, deterministic scoring, valid accounting, task-level analysis, and reproducible outputs.

If all 48 main cells complete, the report may claim **observed differences in controlled delayed scholarly-evidence answer accuracy across four operational architectures** within this 12-case development sample. It may describe mechanism traces and measured token/latency tradeoffs. It **cannot** claim that the result establishes human-level comprehension, broad academic relevance, a reduction in scientific hallucinations, or superiority of Meta's official CLM.

If insufficient cases, missing pressure, graph delivery defects, or state-fork failure occur, state these explicitly and narrow conclusions. Do not label an unverified mechanism as tested.

### Paper presentation (eight-page two-column final target)

Recommended compact allocation for this controlled study: one method diagram (8 batches → context state → two forked question branches), one results figure (paired effects or four-method primary accuracy with N=12), and 1–2 short paragraphs distinguishing **retention** from **recovery** and explaining limitations. The separate natural-retrieval experiment remains the paper's broader effectiveness test.

## 13. Intellectual precedent and attribution

This is a **new, project-specific evaluation protocol**, not a reproduction of an existing published benchmark. For conceptual precedent, LongMemEval (Wu et al., ICLR 2025) distinguishes information extraction, multi-session reasoning, temporal reasoning, knowledge updates and abstention in long-term memory. Our design borrows the *idea of explicit memory capabilities and controlled delayed questions*, but uses eight batches of real OpenAlex papers, citation-grounded questions and a graph × CLM factorial ablation. Cite the original research when writing the paper; do not imply our tasks were part of LongMemEval.

**Original source:** Wu et al., *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*, ICLR 2025, https://proceedings.iclr.cc/paper_files/paper/2025/hash/d813d324dbf0598bbdc9c8e79740ed01-Abstract-Conference.html

---

### Design completion boundary

This file **fully specifies the evaluation design, invariants, primary and secondary endpoints, scoring, statistics, resource controls, and implementation gates**. It does **not** fabricate twelve real scholarly cases or assert that the local repository contains eligible records; those require Codex to materialize and audit from the frozen local corpus. No API calls or empirical Terra runs have been performed to create this specification.
