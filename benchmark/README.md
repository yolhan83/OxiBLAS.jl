# Benchmarks

## High-level suite: `run.jl`

A generated grid: every high-level operation (products, factorizations and
their in-place / refactorization forms, `OxiBLAS.ldiv` / `ldiv!` / `\`, SVD
and spectra, determinants, norms, matrix functions, Sylvester/Lyapunov,
`einsum`) on every input type it accepts:

- **LinearAlgebra types:** `Matrix`, tall `Matrix`, `Hermitian`,
  `UpperTriangular`, `LowerTriangular`, `Diagonal`, `Bidiagonal`,
  `Tridiagonal`, `SymTridiagonal`, `SparseMatrixCSC`;
- **OxiBLAS types:** `CSRMatrix`, `CSCMatrix`, `COOMatrix` and their
  `Hermitian` wrappers, `BandedMatrix`, `PackedMatrix`;
- **element types:** `Float16`, `Float32`, `Float64`, `ComplexF32`,
  `ComplexF64`, `BigFloat`;
- **three sizes** (`s`, `m`, `l`) per input kind, smaller for `Float16` and
  `BigFloat` (no BLAS: LinearAlgebra runs its generic code).

Case names are `op/input/type/size`, for example `lu/csr/f64/m`
(`suite/operations.jl` and `suite/inputs.jl` define the grid). Each case is
compared with the matching reference on the same data and thread count. The
OxiBLAS result is checked by a backward error or a reconstruction, which does
not need the reference. Each case records the allocation count and bytes of
one warm call, on both sides, and the median time of warm samples
(Chairmarks). Status: `ok`, `unchecked` (no way to validate without a
reference), `unsupported` (no OxiBLAS method for the input), `mismatch`,
`error`. Results go to `results/<stamp>.csv` and `.md`, with summaries by
family, operation, element type and input, and an allocation section
(in-place cases that allocate, cases that allocate more than the reference).

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark --threads=1 benchmark/run.jl                       # whole grid (about 4,000 cases)
julia --project=benchmark --threads=1 benchmark/run.jl lu cholesky           # name prefixes
julia --project=benchmark --threads=1 benchmark/run.jl 'ldiv!/*/f64/*'       # glob on op/input/type/size
julia --project=benchmark benchmark/run.jl --types=f32,f64 --sizes=s,m --families=solves --inputs=csr,dense
julia --project=benchmark benchmark/run.jl --allocs                          # allocation campaign, no timing
julia --project=benchmark benchmark/run.jl --list                            # selected case names
OXIBLAS_REFERENCE=accelerate julia --project=benchmark benchmark/run.jl      # macOS
OXIBLAS_BENCH_SECONDS=2 julia --project=benchmark benchmark/run.jl lu        # longer sampling (default 0.5 s)
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
