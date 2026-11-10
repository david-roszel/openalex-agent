# OpenAlex Research Agent

An instrumented scholarly research system with live OpenAlex retrieval, bounded hosted planning, persistent citation graphs, CLM-inspired working-context edits, source-traceable synthesis and a Streamlit evaluation dashboard.

**Research status:** five architectures form the primary comparison. Official Meta CLM remains exploratory. Existing development observations establish implemented mechanisms and important failure modes; they do **not** establish graph/CLM superiority, hallucination reduction, independent factual accuracy or broad generalization. See [results and limitations](docs/RESULTS_AND_LIMITATIONS.md).

## Install

Python 3.11+ is declared; Python 3.14/Linux is locally tested. On Ubuntu, install `python3-venv` and `python3-pip` if necessary.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,ui]'
test -f .env || cp .env.example .env
```

Use your own keys in the ignored `.env` for live research. Saved-output inspection, scripted diagnostics and offline tests need no credentials. The checked-in template contains none. Provider access, current pricing and billing must be verified for your own account; the recorded Terra configuration is not a guarantee of availability.

## Two interfaces

### Live Research: question → investigation → cited answer

```bash
./launch-live-research.sh
```

Open **http://127.0.0.1:8521**. Advanced Settings selects conventional, graph, CLM or combined hosted research, source type, limits and saved runs. A saved run opens without inference. Clicking **Research** starts an individual hosted investigation and can incur charges; it does not authorize a benchmark campaign.

For another port: `OPENALEX_DASHBOARD_PORT=8541 ./launch-live-research.sh`. For an optional recorded-only session: `OPENALEX_DEMO_READ_ONLY=1 ./launch-live-research.sh`; use `0` or unset it for normal live research. No private research trace or grading key is shipped.

### Scientific Evaluation: comparisons and evidence inspection

```bash
OPENALEX_DASHBOARD_PORT=8511 ./launch-dashboard.sh
```

Open **http://127.0.0.1:8511**. The five workspaces are **Scientific Evaluation**, **Individual research**, **Experiment Dashboard**, **Long-Horizon Testing** and **Saved Experiments**. Views include architecture comparisons, timelines, actual graph/context inputs, paper rankings and provenance. The forest/sage/tan/cream theme supports light and dark modes.

Scientific Evaluation reads saved results; it does not launch inference. The checkout starts without private saved experiments. To create a local **synthetic engineering demonstration**, open Experiment Dashboard, choose **Diagnostics**, keep the deterministic planner and click **Run Comparison**. This is mocked/scripted correctness data, not a hosted model result or scientific quality judgment. It can then be inspected through saved-result views. Other execution controls retain their review/approval gates.

See [UI guide](docs/UI_GUIDE.md). The two interfaces share the current implementation; older benchmark behavior is isolated separately rather than silently merged.

## Primary architecture comparison

|Method|Graph-derived state|Working context|Role|
|---|---|---|---|
|`standard`|No hosted graph intervention|No hosted model|Single-pass deterministic reference; no factual synthesis requirement|
|`llm`|Off|Conventional bounded context|Hosted scholarly actions and permitted recovery|
|`llm_graph`|On|Conventional bounded context|Adds graph-derived planner information|
|`clm`|Off|Model-editable bounded text|CLM-inspired editing, archival and recovery|
|`clm_graph`|On|Model-editable bounded text|Combines both interventions|

The four hosted cells form a 2×2 design; standard is an external reference with a different work/output obligation. Shared ceilings do not guarantee equal actual evidence exposure or compute. Simple UI aliases are `conventional`, `graph`, `clm`, `combined`.

`meta_clm` and `meta_clm_graph` retain the official conversation renderer/parser/synchronizer integration as **exploratory** configurations. They are excluded from the primary five-method claims. See [seven-method compatibility](docs/SEVEN_ARCHITECTURES.md) and [CLM details](docs/CLM_IMPLEMENTATION.md).

Canonical OpenAlex IDs prevent duplicate paper identities. Retrieved publications and unresolved citation targets are distinct; directed citations derive from source metadata. A stored graph, a selected graph record, a delivered payload and causal model reliance are separate claims. Quotes/source IDs provide traceability, not independent scientific verification.

## CLI and offline examples

Inspect the profile without network or inference:

```bash
.venv/bin/python -m openalex_agent.live_cli 'How do iterative retrieval and graph retrieval differ?' --dry-run
```

A deterministic run over39 small authentic historical metadata records:

```bash
.venv/bin/python -m openalex_agent.live_cli \
  'How can citation networks support scholarly literature discovery?' \
  --planner deterministic --snapshot tests/fixtures/development-corpus --output data/evidence-digest.md
```

This emits an evidence digest, not an LLM answer or new scientific finding. The separately retained `tests/fixtures/synthetic-corpus` and generic `benchmarks/example_*.json` are synthetic examples. `benchmarks/live_submission_v1/questions.json` is a fixed, development-only question/target manifest required by the live runner and tests; it is not a sealed held-out set or human relevance annotation.

The scripted end-to-end mechanism exercise also uses MockTransport and labels its output:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=integration/offline_guard:. \
  .venv/bin/python scripts/exercise_submission_offline.py
```

Its saved-run artifacts stay under ignored `data/`; its generated receipt under `docs/submission/` is also ignored. Both are engineering diagnostics. The script's nested output location is not automatically the top-level UI's saved-run list. For an in-UI offline example use Diagnostics above.

## Limits and accounting

The current Live Research profile defaults to18 actions/rounds,144 admitted records,40 metadata attempts,28 all-stage hosted calls,150,000 total tokens,24,000 payload characters,600 seconds and a$1.50 individual inference ceiling. Advanced Settings can raise the visible time/spending selections to1,800 seconds/$20. These are editable per-run limits, not a cumulative experiment authorization or a quality guarantee.

Durable request reservations precede hosted transmission. Unaccountable/interrupted usage retains holds and blocks further paid inference; duplicate request keys are not replayed. A bounded content revision is distinct from retrying a failed HTTP request. OpenAlex has its own request/observed-cost guards; missing monetary reports remain unknown, and observed cost is not a prepaid hard cap. Authentication, quota, service failures and partial answers remain visible.

Pricing in `.env.example` records the source's conservative development profile. No free-credit eligibility, current account rate or live key access is assumed. [Configuration](docs/CONFIGURATION.md) explains the supported variables and optional Gemini backend.

## V2/V3 and Revision7 research implementation

Useful frozen benchmark functionality is preserved under **[versions/revision7](versions/revision7/README.md)**: actual directed graph-boundary accounting, item-level null abstentions, required-final classification, acquisition/inspection separation, progressive exposure measurement and atomic detached worker/accounting infrastructure. Its runtime contains the earlier V2 graph machinery.

This package has its own `openalex_agent` namespace and historical policies. Run its synthetic tests in its directory/in a separate Python process. Do not install both versions into one editable environment or replace current application modules with frozen ones. No approvals, live ledgers, full corpus, private case/gold files or original hosted traces are released. A full scientific rerun is therefore not claimed.

The optional [legacy progress monitor](tools/legacy-monitor/README.md) supports the historical 900-cell schema without provider calls; it is not a universal R7 monitor.

[Iteration decisions](docs/ITERATIONS.md) explains what was retained, superseded or left private. Git history preserves the earlier public release. Frozen source hashes identify benchmark code independently from the current app.

## Testing, reproducibility and publication

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=integration/offline_guard:. \
  .venv/bin/python -m pytest -p no:cacheprovider
./scripts/test-revision7-offline.sh
.venv/bin/python scripts/repository_audit.py
.venv/bin/python scripts/verify_manifest.py
```

Root and frozen tests use synthetic/mocked providers; the inherited offline guard blocks real IP connections. These tests validate software contracts, not scholarly effectiveness. The optional Gemini SDK has separate dependencies; related tests may skip when it is absent. CI declares Python 3.11/3.14, but only the locally executed environment is reported as passed until CI runs.

[Observed validation](docs/VALIDATION.md), [reproducibility](docs/REPRODUCIBILITY.md), [evaluation policy](docs/EVALUATION.md), [five-method development collection](docs/FIVE_METHOD_DEVELOPMENT.md) and [pooled review](docs/POOLED_RELEVANCE_PROTOCOL.md) describe versioning and independent judgments. Relevance grades remain0–2. Missing judgments stay NOT EVALUATED; final held-out and paid collection require their existing explicit approvals. No approvals or human annotations are fabricated or shipped.

`PUBLIC_MANIFEST.json` records the prepared source inventory; manifests and Git commits must both identify later experiments. Source hashes change with this update; historical runs retain their original identities. [PUBLISHING.md](PUBLISHING.md) provides the reviewed commit/push procedure. The recorded release `f76e842f` is committed on `origin/main`; this follow-up security/documentation cleanup is uncommitted. Repository visibility remains private pending owner action. Preparation does not commit or publish automatically.

## Data, credentials and licenses

Only small fixture metadata, development/example inputs and an aggregate, qualified research-status summary are included. No30k corpus, private grading key/package, live ledger, held-out questions, human annotations, workstation state or raw hosted transcripts are included. Internal milestone/audit documents are distilled into public technical/version/limitations documentation rather than copied wholesale.

Original project code has no project-wide license grant yet; public visibility does not imply unrestricted reuse. Vendored Meta components retain CC BY-NC 4.0, their NOTICE and pinned provenance, including noncommercial restrictions. Tokenizer assets retain their MIT/OpenAI attribution. See [licensing boundaries and owner decisions](docs/LICENSING.md).

The [November 10 security review](docs/SECURITY_REVIEW.md) covers the recorded release's complete enumerated Git history and locally stored unreachable content. No confirmed credentials were found; historical contact/path metadata and licensing decisions remain owner considerations. Current publication guards supplement that audit; neither tests nor a secret scan establishes scientific superiority or rules out every security vulnerability.
