#!/usr/bin/env bash
# PREPARED ONLY. Each phase requires explicit authorization of this campaign.
# Usage: bash benchmark/archive/run_precision_campaign.sh NEW_DIR correctness|performance|rust|inspect
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
output="${1:?new output directory required}"
phase="${2:?explicit phase required}"
case "$phase" in correctness|performance|rust|inspect) ;; *) exit 2 ;; esac
[[ ! -e "$output" ]] || { echo 'Refusing to overwrite a campaign' >&2; exit 2; }
mkdir -p "$output"
output="$(cd "$output" && pwd)"
cd "$root"
# Optional providers use a separate, explicitly prepared test/optional environment.
correctness_project="${OXIBLAS_CORRECTNESS_PROJECT:-.}"
precision_project="${OXIBLAS_PRECISION_PROJECT:-benchmark}"
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
        run_logged load julia --project="$correctness_project" --threads=1 --check-bounds=yes -e 'using OxiBLAS, InteractiveUtils; versioninfo(verbose=true)'
        for threads in 1 2 4 8; do
            run_logged "tests-$threads" julia --project="$correctness_project" --threads="$threads" --check-bounds=yes test/legacy/run_extended.jl
        done
        ;;
    performance)
        for threads in 1 2 4 8; do
            run_logged "native-$threads" julia --check-bounds=auto --project="$precision_project" --threads="$threads" benchmark/archive/precision_complete.jl "$output/native-$threads" native
            run_logged "reference-$threads" julia --check-bounds=auto --project="$precision_project" --threads="$threads" benchmark/archive/precision_complete.jl "$output/reference-$threads" reference openblas
        done
        ;;
    rust)
        run_logged rust-version rustc --version --verbose
        # Offline build may create its Cargo.lock; generated binaries stay in
        # this new campaign directory. Missing cached dependencies are failures.
        run_logged rust-build cargo build --offline --release --manifest-path benchmark/rust_precision_reference/Cargo.toml --target-dir "$output/rust-target"
        binary="$output/rust-target/release/oxiblas-precision-reference"
        run_logged rust-probe "$binary" probe "$output/rust-probe.csv"
        run_logged rust-oracle julia --check-bounds=yes --project="$precision_project" --threads=1 benchmark/archive/compare_extended_reference.jl "$output/rust-probe.csv"
        run_logged rust-timing "$binary" bench "$output/rust-timing.csv"
        shasum -a 256 "$output/rust-probe.csv" "$output/rust-timing.csv" > "$output/rust-results.sha256"
        ;;
    inspect)
        run_logged extended-inspect julia --check-bounds=auto --project="$precision_project" --threads=1 benchmark/archive/inspect_extended.jl "$output/ir"
        ;;
esac
