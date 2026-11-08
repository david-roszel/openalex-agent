# Comparative dashboard

Launch `./launch-dashboard.sh` after installing `.[dev,ui]`. It uses `.venv`, binds
to localhost and defaults to port 8501; `OPENALEX_DASHBOARD_PORT` selects another.
The direct alternative is `python -m streamlit run app.py`.

Experiment Dashboard offers selected-method, graph-ablation, CLM-ablation,
factual-evaluation, five-method and G1–G5/C1–C7 correctness runs. Diagnostics
are offline scripted fixtures. A complete 2×2 agent comparison requires all four
agent cells; subsets support conditional contrasts only. Fixed diagnostic test
selections run their existing configurations rather than rewriting those fixtures
from the comparison controls.

Use Run Comparison once after reviewing shared settings, budgets and model
configuration. Isolated states are created for each method/question/trial.
Reruns reuse the submission; Prepare another experiment explicitly creates a new
one. Polling reads committed files/events. Cancellation is cooperative: an
in-flight request can finish first, pressure stops between arms, and partial
states/errors remain available. General interrupted-job resume is not provided.

Results show per-question/trial metrics, available descriptive plots, raw traces,
context snapshots and configuration. Unknown labels yield NOT EVALUATED. Context
timelines show actual inputs and requested/committed/failed edits; quote presence
is not semantic support. Graph inspection separates planning inputs/actions from
the common final lexical rankings. Saved Experiments can inspect several runs and
export summaries/figures without model calls.

Long-Horizon Testing configures measured pressure, evidence/distractor placement,
correction and archive access. Invalid pressure conditions remain raw evidence
and are excluded from aggregate displays. Scripted usage is not billed provider
usage. Empirical development/final modes require independently reviewed corpora,
public probes/labels and pinned coordinator approval. The released synthetic
examples are correctness inputs, not valid final-study material.

Reviewed bundles are discovered from local `data/*/review-manifest.json`.
Scientific input files are kept under approved local experiment locations.
No actual human review bundle, approval or empirical corpus is distributed.
This localhost, single-process tool has no multiuser authentication, distributed
queue or account-wide cost coordination.

The existing dashboard controls remain five-method controls. Seven-method runs
use the benchmark CLI or separately pinned provider runner; their implementation
is described in [seven-method integration](SEVEN_ARCHITECTURES.md).
