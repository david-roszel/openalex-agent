#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
exec "$project_dir/.venv/bin/python" -m streamlit run "$project_dir/app.py" \
  --server.address 127.0.0.1 --server.port "${OPENALEX_DASHBOARD_PORT:-8501}" \
  --server.headless true --browser.gatherUsageStats false
