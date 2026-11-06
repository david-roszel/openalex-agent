# OpenAlex / CLM Research Agent

Research software for comparing graph memory and model-controlled working
context in academic literature retrieval. It includes an OpenAlex client,
incremental knowledge graph, bounded LLM planning, a controlled five-method
benchmark, evidence assessment and a local Streamlit dashboard.

**Research status:** the implementation and correctness diagnostics are available.
This release does not establish improvements in retrieval quality, factual
accuracy, retention, latency or cost from CLM or graph memory. No final judged
held-out evaluation or independent human annotations are distributed.

## Architecture and configurations

| Method | Graph memory | Working context | Planning |
| --- | --- | --- | --- |
| `standard` | No | Single-search baseline | No LLM |
| `llm` | No | Environment-managed bounded context | Hosted LLM |
| `llm_graph` | Yes | Environment-managed bounded context | Hosted LLM |
| `clm` | No | Model-editable bounded working text | Hosted LLM |
| `clm_graph` | Yes | Model-editable bounded working text | Hosted LLM |

The four agent cells form a 2×2 graph/context design. They share retrieval tools,
corpus access, validation rules and final lexical ranking. Graph planning inputs
and context-management behavior intentionally differ. All methods start with
isolated state. Select `--suite controlled-five`; the older CLI `legacy` suite is
a separate compatibility interface, not this factorial design.

OpenAlex metadata enters the retrieval/cache layer, then observed records enter
agent memory and optionally the graph. The planner chooses validated actions;
the controller applies budgets and commits observations. CLM-inspired agents
can edit working text and recover archived observations. Metadata, graph records
and raw evidence remain authoritative outside that editable text.

## Install and test

Python 3.11+ is declared. Local release verification uses Python 3.14; other
interpreters and the GitHub Actions matrix require their own CI verification.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,ui]'
python -m pytest
```

Dependencies are declared in `pyproject.toml`. `requirements.txt` installs the
core development extras. `requirements-ui.lock.txt` captures a Python 3.14 UI/dev
environment; install it followed by `python -m pip install --no-deps -e .` when
that environment is appropriate. Lock snapshots contain version pins, not
cross-platform wheel archives or package integrity hashes.

Offline tests require no credentials, network APIs or private data. The bundled
fixtures are explicitly synthetic. Tokenizer ranks and their upstream MIT
license are packaged to avoid tokenizer downloads during tests.

For live use, copy `.env.example` to a local `.env` and enter your own OpenAlex
key and HTTPS hosted-planner endpoint/key/model. Provider/model access and request
parameters depend on your account. No model is recommended or preconfigured in
this release. See [configuration](docs/CONFIGURATION.md). Live requests may incur
charges; offline commands below use deterministic policies and local snapshots.

## Launch the dashboard

```bash
./launch-dashboard.sh
# Use another port when needed:
OPENALEX_DASHBOARD_PORT=8502 ./launch-dashboard.sh
```

Open <http://localhost:8501> (or the chosen port). `python -m streamlit run app.py`
is the direct alternative. The launcher binds to localhost and disables usage
telemetry. The four workspaces are Experiment Dashboard, Long-Horizon Testing,
Saved Experiments and Individual research.

Start with **Diagnostics → Comparison → Run Comparison**. Select all five
methods, or a graph/CLM pair, inspect shared settings and compare actual recorded
outputs. Diagnostics use scripted models and synthetic evidence. Context views
show requests and committed edits; graph views show planning inputs and actions.
Cancellation preserves partial artifacts. Saved runs can be inspected without
new calls. Missing judgments display NOT EVALUATED; failed pressure conditions
are labeled INVALID PRESSURE CONDITION and excluded from aggregate displays.

Empirical modes require user-supplied, independently reviewed data and pinned
protocol approval. Paid confirmation does not bypass readiness. No private
question sets, annotations, approval files or past experiments are included.
See [dashboard usage](docs/DASHBOARD.md) and [evaluation](docs/EVALUATION.md).

## Run the agent and reproduce an engineering benchmark

An offline agent loop over synthetic records:

```bash
python main.py agent "synthetic scholarly network examples" \
  --planner deterministic --corpus tests/fixtures/synthetic-corpus \
  --max-actions 3 --max-rounds 3 --max-api-requests 1
```

A complete mocked-provider demonstration:

```bash
python -m openalex_agent.agent_demo --output data/examples/agent-demo
```

A five-method engineering sweep using only public synthetic inputs:

```bash
python main.py benchmark --suite controlled-five \
  --corpus tests/fixtures/synthetic-corpus \
  --questions benchmarks/example_questions.json \
  --planner deterministic --allow-deterministic \
  --methods standard llm llm_graph clm clm_graph \
  --action-budgets 3 --trials 1 --trial-seeds 0 \
  --max-rounds 3 --max-inspected 12 --page-size 2 --max-api-requests 1 \
  --max-llm-calls 5 --max-total-tokens 70000 \
  --active-context-tokens 4096 --input-tokens 8192 \
  --output data/examples/five-method-smoke
```

Use fresh output directories. A positive request ceiling is required by the
existing controller, but fixed-corpus runs make no live OpenAlex requests.
Deterministic agents are engineering policies, not LLM-quality baselines.
For graph/context mechanisms and pressure:

```bash
python -m openalex_agent.controlled_diagnostics --output data/examples/diagnostics
python -m openalex_agent.context_pressure --condition low --output data/examples/pressure
```

Actual live search is separate and requires your own configuration:

```bash
python main.py search "academic literature retrieval" --limit 5
```

See [reproducibility](docs/REPRODUCIBILITY.md) for manifests, scoring and artifact
handling, and `python main.py --help` for the complete CLI.

## Scientific limitations

This is a CLM-inspired context-as-a-file adaptation, not a trained memory model
or a reproduction of a published training/evaluation harness. It provides bounded
JSON text edits and observation recovery, not arbitrary shell access, hidden
reasoning, native provider-context control or compute-cache reuse.

Scripted retention/correction outcomes validate mechanisms only. Lexical ranking
can miss terminology; citation expansion can drift; abstracts may not support a
claim; valid JSON does not ensure a good decision. Study-token counts are
separate from provider billing tokens. Hosted responses may vary even with the
same inputs. Resource ceilings do not guarantee exact provider invoices.

Independent relevance judgments, verified factual support, repeated trials and a
frozen held-out protocol are required before drawing comparative conclusions.
Future studies may show no improvement or regressions. No independent human
judgments or empirical results have been invented for this release.

## Release contents

Technical design: [algorithms](docs/ALGORITHMS.md), [graph](docs/GRAPH_DESIGN.md),
[CLM adaptation](docs/CLM_IMPLEMENTATION.md), [evaluation](docs/EVALUATION.md).
The included tests cover core retrieval, graph correctness, planning, controlled
comparisons, evidence accounting and Streamlit rendering. Historical provider
pilot wrappers and their dedicated tests are outside the release scope.

Internal milestones, drafts, audits, unpublished evaluation questions, private
annotations, local data, API keys and Git history are excluded. Generated outputs
belong under ignored `data/`. Run `python scripts/repository_audit.py` before
staging; `--staged` checks exact index blobs after you initialize Git. See
[PUBLISHING.md](PUBLISHING.md) for commands; no initialization or publication has
been performed automatically.

`PUBLIC_MANIFEST.json` identifies included files and hashes. The complete private
selection/exclusion review is stored separately, outside this export. The
packaged tokenizer retains its upstream MIT license. A project-wide code license
and author/citation metadata remain owner decisions; none has been invented.
