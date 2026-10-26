#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export TECTONIC_CACHE_DIR="$project_root/.cache/tectonic"

python "scripts/build_report_figures.py"
tectonic --keep-logs --outdir reports/ieee reports/ieee/main.tex
# Install reports/requirements.txt first. Both deck formats use one layout.
python "scripts/build_presentation.py"
