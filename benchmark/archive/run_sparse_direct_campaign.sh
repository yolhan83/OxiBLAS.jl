#!/usr/bin/env bash
# PREPARED ONLY. Execute only after explicit authorization of this exact campaign.
# Usage: bash benchmark/archive/run_sparse_direct_campaign.sh NEW_DIRECTORY [correctness|performance]
# Prerequisite: dependencies already prepared in an independently authorized step.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
output="${1:?new output directory required}"
phase="${2:-correctness}"
case "$phase" in correctness|performance) ;; *) exit 2 ;; esac
[[ ! -e "$output" ]] || { echo 'Refusing to overwrite a campaign' >&2; exit 2; }
mkdir -p "$output"
output="$(cd "$output" && pwd)"
cd "$root"
run_logged() {
    local name="$1"
    shift
    date -u '+%Y-%m-%dT%H:%M:%SZ' > "$output/$name.start"
    local status=0
    "$@" > "$output/$name.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "$output/$name.exit"
    date -u '+%Y-%m-%dT%H:%M:%SZ' > "$output/$name.end"
    shasum -a 256 "$output/$name.log" > "$output/$name.log.sha256"
    [[ "$status" -eq 0 ]] || { cat "$output/$name.log" >&2; return "$status"; }
}
if [[ "$phase" == correctness ]]; then
    run_logged load julia --project=. --threads=1 --check-bounds=yes -e 'using OxiBLAS, InteractiveUtils; versioninfo(verbose=true)'
    for threads in 1 2 4 8; do
        run_logged "tests-$threads" julia --project=. --threads="$threads" --check-bounds=yes test/legacy/run_sparse_direct.jl
    done
else
    # Correctness campaign must first be reviewed. This phase has a separate
    # output directory and is never automatically started by correctness.
    for threads in 1 2 4 8; do
        run_logged "native-$threads" julia --check-bounds=auto --project=benchmark --threads="$threads" benchmark/archive/sparse_direct_complete.jl "$output/native-$threads" native
        run_logged "reference-$threads" julia --check-bounds=auto --project=benchmark --threads="$threads" benchmark/archive/sparse_direct_complete.jl "$output/reference-$threads" reference openblas
    done
fi
