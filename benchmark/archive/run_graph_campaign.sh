#!/usr/bin/env bash
# PREPARED ONLY. Every phase requires explicit inclusion in an approved campaign.
# Usage: bash benchmark/archive/run_graph_campaign.sh NEW_DIR correctness|performance|rust|inspect
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
output="${1:?new output directory required}"
phase="${2:?explicit phase required}"
case "$phase" in correctness|performance|rust|inspect) ;; *) exit 2 ;; esac
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
case "$phase" in
    correctness)
        run_logged load julia --project=. --threads=1 --check-bounds=yes -e 'using OxiBLAS, InteractiveUtils; versioninfo(verbose=true)'
        for threads in 1 2 4 8; do
            run_logged "tests-$threads" julia --project=. --threads="$threads" --check-bounds=yes test/legacy/run_graphs.jl
        done
        ;;
    performance)
        for threads in 1 2 4 8; do
            run_logged "graphs-$threads" julia --check-bounds=auto --project=benchmark --threads="$threads" benchmark/archive/sparse_graphs.jl "$output/graphs-$threads"
        done
        ;;
    rust)
        # No network or implicit package installation. An unavailable offline
        # dependency is a recorded failure, never an inferred reference result.
        run_logged rust-version rustc --version --verbose
        run_logged rust-build cargo build --offline --release --manifest-path benchmark/rust_graph_reference/Cargo.toml --target-dir "$output/rust-target"
        run_logged rust-benchmark env RAYON_NUM_THREADS=1 "$output/rust-target/release/oxiblas-graph-reference" "$output/rust.csv"
        shasum -a 256 "$output/rust.csv" > "$output/rust.csv.sha256"
        ;;
    inspect)
        run_logged graph-inspect julia --check-bounds=auto --project=benchmark --threads=1 benchmark/archive/inspect_graphs.jl "$output/ir"
        ;;
esac
