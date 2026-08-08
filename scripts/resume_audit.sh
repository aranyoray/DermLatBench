#!/usr/bin/env bash
# resume_audit.sh — same as run_audit.sh; the pipeline is idempotent.
set -Eeuo pipefail

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

exec dermlat-audit resume "$@"
