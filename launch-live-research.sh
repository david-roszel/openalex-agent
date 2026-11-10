#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
# Normal use is interactive research. The recording-only restriction requires
# an explicit OPENALEX_DEMO_READ_ONLY=1; it is never the launch default.
export OPENALEX_DEMO_READ_ONLY="${OPENALEX_DEMO_READ_ONLY:-0}"
python_bin="${OPENALEX_PYTHON:-$project_dir/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then python_bin=python3; fi
exec "$python_bin" -m streamlit run "$project_dir/live_app.py" \
  --server.address 127.0.0.1 --server.port "${OPENALEX_DASHBOARD_PORT:-8521}" \
  --server.headless true --browser.gatherUsageStats false
