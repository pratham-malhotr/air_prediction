#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export MPLCONFIGDIR="$project_root/.cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1

if [[ -x "$project_root/.venv/bin/python" ]]; then
  project_python="$project_root/.venv/bin/python"
elif [[ -x /opt/anaconda3/bin/python ]]; then
  project_python=/opt/anaconda3/bin/python
else
  project_python=python3
fi

exec "$project_python" "$project_root/scripts/serve_frontend.py" "$@"
