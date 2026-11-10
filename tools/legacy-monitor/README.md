# Optional read-only historical campaign monitor

This existing standard-library utility displays terminal/HTML progress, failures, per-method/action-budget counts, conservative ledger exposure and estimated remaining time. It snapshots queue.sqlite plus WAL into a temporary directory before querying, avoiding writes to the original DB/SHM. It makes no model/OpenAlex/Codex calls and cannot start, resume, stop or reconcile a benchmark.

It supports the **historical objective campaign schema only**: B-main/C-long, five methods,900 planned main cells. Do not point it at R7's636-cell or LiveDemo's45-cell schema and interpret its counts as valid. No original queue, spending ledger or experimental output is supplied.

From repository root, with your own compatible saved campaign:

```bash
python3 tools/legacy-monitor/monitor.py \
  --benchmark /path/to/compatible/saved-campaign \
  --output-dir data/legacy-monitor-output --once
```

Omit --once for30-second refresh. Open data/legacy-monitor-output/dashboard.html for the auto-refreshing local page. Output must be outside the benchmark input directory. The default input/output paths are relative to ignored repository data/, rather than a developer workstation. No count/scoring/accounting logic was changed for publication. Conservative retained exposure includes reservations/unknown usage and is not an invoice or promotional balance; ETA is an estimate.

Independent synthetic tests, with no original records:

```bash
cd tools/legacy-monitor
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../../integration/offline_guard:. \
  ../../.venv/bin/python -m pytest -p no:cacheprovider test_monitor.py
```
