#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export NO_UPDATE_NOTIFIER=1
export VERCEL_TELEMETRY_DISABLED=1
export PYTHONDONTWRITEBYTECODE=1

if [[ -x /opt/anaconda3/bin/python ]]; then
  project_python=/opt/anaconda3/bin/python
else
  project_python=python3
fi
"$project_python" "$project_root/scripts/prepare_vercel.py"

if command -v vercel >/dev/null 2>&1; then
  exec vercel --cwd "$project_root/dist/aircast-pm25" deploy --prod --yes
fi
shopt -s nullglob
cached_cli=("${XDG_CACHE_HOME:-$HOME/.npm}/_npx/"*/node_modules/vercel/dist/index.js)
if (( ${#cached_cli[@]} == 0 )); then
  cached_cli=("$HOME/.npm/_npx/"*/node_modules/vercel/dist/index.js)
fi
if (( ${#cached_cli[@]} > 0 )); then
  exec node --no-experimental-webstorage "${cached_cli[0]}" --cwd "$project_root/dist/aircast-pm25" deploy --prod --yes
fi
exec npx --yes vercel@latest --cwd "$project_root/dist/aircast-pm25" deploy --prod --yes
