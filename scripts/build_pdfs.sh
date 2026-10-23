#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export TECTONIC_CACHE_DIR="$project_root/.cache/tectonic"

tectonic --keep-logs --outdir reports/ieee reports/ieee/main.tex
tectonic --keep-logs --outdir reports/presentation reports/presentation/slides.tex
