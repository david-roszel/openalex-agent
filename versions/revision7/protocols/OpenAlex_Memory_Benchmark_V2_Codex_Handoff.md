# Codex handoff — implement the frozen OpenAlex Controlled Scholarly Memory Benchmark V2

**Use Codex High. Do not make paid calls during this implementation prompt.**

The attached bundle contains:

1. `OpenAlex_Controlled_Memory_Benchmark_V2_Protocol.md` — complete, prespecified experimental design; treat as the normative source.
2. `OpenAlex_Memory_Case_Manifest_Schema_V2.json` — machine-readable structure of each fixed 24-paper case.
3. `OpenAlex_Memory_Answer_Schema_V2.json` — exact six-question final answer contract.
4. `openalex_memory_scoring_v2.py` — tested, deterministic Python reference scorer for answer accuracy and factorial effects.

## Instructions

Work in the existing OpenAlex agent integration repository. Inspect the existing eight-batch/three-paper memory-probe infrastructure and reuse it where feasible. Implement a **separate** controlled-memory V2 runner and freeze it before any scored Terra calls.

Your task now is **engineering and offline case materialization only**:

- Materialize 12 real scholarly cases, two from each of the six existing domain labels, by deterministic selection of 24 distinct real papers per case from the frozen OpenAlex corpus. Each case must include audited directed B→A and C→B citation relationships. Do not invent papers, edges, years or passages. Freeze canonical IDs, the eight batches of three, exact source excerpts, questions, answer keys, corpus hashes and selection rules.
- Materialize two **disjoint** pilot cases for later mechanical validation, not included in the 12 main cases.
- Use the case-manifest schema and write withheld gold keys to a private file inaccessible to the agent; construct questions deterministically from actual evidence, and audit each answer for uniqueness before model calls.
- Ensure the same 24 real-paper exposures are delivered, in the same order, to each of the four hosted methods: `llm`, `llm_graph`, `clm`, `clm_graph`.
- Enforce exactly the same active/full context, character, provider-call, and recovery budgets across these four methods. Verify genuine finite-context pressure from measured serialized inputs; do not simply claim eight batches are long.
- Verify explicit citation-edge delivery at the **actual provider-input boundary** for the graph arms. Both graph arms must receive equivalent legitimately available edge data. They may reorganize only metadata actually exposed to all methods, with no hidden key/corpus lookahead.
- Ensure both CLM-inspired arms can commit an edit and preserve its result into the next true model request, including graph-derived evidence in the combined arm. No forced editing in natural or controlled scored cases.
- Implement the immutable post-batch-8 state snapshot and fork into two fully independent final-answer branches: (A) zero external evidence recovery and (B) up to two common evidence-recovery operations. The final six questions are only revealed after the full exposure stream. Do not allow answers from one branch into the other.
- Use the supplied scorer (or a byte-for-byte equivalent independently tested implementation), producing Q1–Q6 correctness and the primary Q1–Q5 mean without an LLM judge.
- Implement case-level paired factorial contrasts and uncertainty analysis. Twelve *cases* are the independent units, not 48 trajectories or individual questions. Report failed, missing, and exposure-incomplete cells separately.
- Reuse the existing **single cumulative $68 gross Terra campaign cap** shared with the natural retrieval experiments. Do not instantiate a new $68 allowance for these probes. Preserve all historical Google artifacts and financial unknowns separately; no Gemini retries or original-campaign restart.
- Connect outputs to the read-only dashboard, keeping the natural and controlled experiment results separate.
- Run offline integration tests with genuine frozen-source fixtures and fake model receipts, including adverse cases, malformed output, missing exposure, wrong edge direction, score correctness, and branch isolation.
- Produce a study readme, frozen manifests and hashed answer keys; the full planned 12×4 matrix; expected provider-call and conservative cost envelopes using locally verified Terra prices; and exact commands for a later, **explicitly authorized** paid pilot and main execution.

## Non-negotiable review gates

1. Main/pilot case and source overlap audit.
2. Graph delivery audit from final serialized requests.
3. CLM persistence canary.
4. Per-batch capacity and genuine pressure audit.
5. No-leakage audit for withheld answer keys and future papers.
6. Snapshot-fork independence test.
7. Deterministic scorer + complete-case/ITT analysis tests.
8. Global $68 reservation/receipt accounting tests.

If any critical gate fails, do not launch inference or invent substitute results. Report the exact issue and proposed fix. This is an objective-development study, not an independent held-out human evaluation or a full hallucination benchmark.

## Final deliverables

- Paths to new implementation modules, manifests, snapshots, protocols and test results.
- Verified record/edge and question-count totals by domain.
- Evidence the graph and CLM mechanisms work at the actual model boundary.
- Verified memory pressure per case (tokens and characters).
- Offline scorer results and statistical-plan verification.
- Provider-call and spending feasibility versus the remaining **shared $68 cap**.
- Exact paid pilot and main run commands, **but do not run them yet**.

Do not perform a five-perspective council review; the user explicitly does not need it for this task.
