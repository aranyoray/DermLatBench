#!/usr/bin/env bash
# run_audit.sh — start (or continue) an audit run.
set -Eeuo pipefail

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

exec dermlat-audit run "$@"
