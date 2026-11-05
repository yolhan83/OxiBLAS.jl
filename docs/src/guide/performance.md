# Performance and benchmarks

## High-level suite

`benchmark/run.jl` measures the operations users call, against the matching
reference:

| Family | Reference |
|---|---|
| GEMM, `OxiBLAS.mul!`, high-level BLAS | OpenBLAS (`LinearAlgebra`), Accelerate on macOS |
| LU, Cholesky, QR, SVD, eigenvalues | OpenBLAS LAPACK, Accelerate |
| `A \ b`, `OxiBLAS.ldiv`, `ldiv!` | the same problem with `LinearAlgebra` (dense) or SparseArrays (sparse) |
| sparse factorizations and products | SparseArrays (UMFPACK, CHOLMOD) |
| `einsum` | TensorOperations |
| structured matrices | structured types of `LinearAlgebra` |

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark --threads=1 benchmark/run.jl              # every case
julia --project=benchmark --threads=1 benchmark/run.jl gemm lu       # selection
OXIBLAS_REFERENCE=accelerate julia --project=benchmark benchmark/run.jl
```

**Method.**

- **Same conditions.** Same mathematical problem, same data and same number of
  threads for OxiBLAS and the reference.
- **Verification.** Each result is compared with the reference before timing.
- **Warm timings.** A first call compiles and warms up; the reported times are
  medians of warm samples (Chairmarks).
- **Separate allocation.** Allocating forms (`OxiBLAS.lu(A)`, `A \ b`) and
  in-place forms (`ldiv!`, `gemm!` with a workspace) are distinct cases.
- **Targeted compilation.** Only the selected cases are compiled.
- **Outputs.** Median time, allocated bytes and OxiBLAS/reference ratio, in
  CSV and Markdown under `benchmark/results/`.

The benchmarks are not run by the CI: timings on GitHub runners are not
stable enough.

## Historical campaigns

The kernel micro-benchmarks and full campaigns (BLAS 1/2/3, LAPACK, dense SVD,
sparse, tensors, graphs) are archived in `benchmark/archive/`. They remain
usable for future optimization work; their campaign reports are not part of
the repository.

## Latency and precompilation

The package precompiles, with PrecompileTools, a small workload of common
paths:

- `Float32` and `Float64` GEMM;
- dense LU and Cholesky;
- a sparse factorization and its solve;
- `OxiBLAS.ldiv` and `\` on a `CSRMatrix`;
- an `einsum` contraction.

The first call of these paths therefore no longer pays the compilation. Other
types and variants compile on first use.

Measurements on an Apple M5 Max, Julia 1.14, each path in a fresh process
(`benchmark/latency/run_all.jl`, results in `benchmark/latency/latency_*.tsv`):

| First call | Without workload | With workload |
|---|---|---|
| `gemm!` Float64 | 0.38 s | 0.09 s |
| `OxiBLAS.lu(A) \ b` | 2.85 s | 0.10 s |
| `OxiBLAS.cholesky` | 2.74 s | 0.01 s |
| `OxiBLAS.ldiv(A, b)` | 13.5 s | 0.13 s |
| sparse `A \ b` (Cholesky, LU) | 7.2 s | 0.005 s |
| `einsum` | 3.2 s | 0.06 s |
| `using OxiBLAS` | 0.03 s | 0.15 s |

**Cost of the workload.** Package precompilation goes from about 3 s to 20 s,
and its cache from 10 MB to about 50 MB.

**Disabling the workload.** Add `precompile_workload = false` to the
`[OxiBLAS]` section of `LocalPreferences.toml`.
