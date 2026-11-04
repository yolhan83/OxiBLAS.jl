# Benchmarks

## High-level suite: `run.jl`

The operations users call (GEMM, `OxiBLAS.mul!`, BLAS 3, dense and sparse
factorizations, SVD and eigenvalues, `\` / `OxiBLAS.ldiv` / `ldiv!`, `einsum`,
structured matrices). Each is compared with the matching reference on the
same data and thread count, checked before timing, and measured with
Chairmarks (median of warm samples). Results go to `results/<stamp>.csv` and
`.md`.

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark --threads=1 benchmark/run.jl            # all cases
julia --project=benchmark --threads=1 benchmark/run.jl gemm sparse  # name prefixes
OXIBLAS_REFERENCE=accelerate julia --project=benchmark benchmark/run.jl   # macOS
OXIBLAS_BENCH_SECONDS=2 julia --project=benchmark benchmark/run.jl lu    # longer sampling
```

The CI never runs benchmarks: GitHub runner timings are not stable enough.

## Archived campaigns: `archive/`

Kernel micro-benchmarks and full campaigns (BLAS 1/2/3, LAPACK, dense SVD,
sparse direct, sparse tensors, graphs, precision, Core.Box). Kept for future
optimization work; their campaign reports stay local (`reports/` is not part
of the repository). They are run from the
repository root, for example
`julia --project=benchmark benchmark/archive/level3.jl`. The Rust reference
programs they once compared against are not part of the package; the scripts
that needed them skip or fail on their Rust columns.
