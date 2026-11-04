#!/usr/bin/env bash
# PREPARATION ONLY: run this file only after explicit user authorization.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root" || exit 1
mkdir -p reports || exit 1
stamp="$(date '+%Y%m%d_%H%M%S')"
report="reports/full_run_${stamp}_$$.log"
# One persistent report. All intermediate files/builds use a temporary directory.
# noclobber also protects historical reports when two launches coincide.
(set -o noclobber; : > "$report") || exit 1
printf 'Report: %s/%s\n' "$root" "$report"
python3 -u benchmark/archive/full_report.py "$@" 2>&1 | tee -a "$report"
statuses=("${PIPESTATUS[@]}")
if [[ "${statuses[1]}" -ne 0 ]]; then
    printf 'ERROR: tee failed; the report may be incomplete.\n' >&2
    exit "${statuses[1]}"
fi
exit "${statuses[0]}"
