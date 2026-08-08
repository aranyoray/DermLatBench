#!/usr/bin/env bash
# generate_report.sh — rebuild metrics.json, all CSVs, audit_report.md from the DB.
set -Eeuo pipefail

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

exec dermlat-audit report "$@"
