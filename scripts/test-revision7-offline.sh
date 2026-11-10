#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir/versions/revision7"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$project_dir/integration/offline_guard:$PWD"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/openalex-revision7-matplotlib"
exec "$project_dir/.venv/bin/python" -m pytest -p no:cacheprovider tests "$@"
