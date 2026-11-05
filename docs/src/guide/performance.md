# Performance and benchmarks

## High-level suite

`benchmark/run.jl` measures a generated grid: each high-level operation, on
each input type it accepts, for `Float16`, `Float32`, `Float64`, `ComplexF32`,
`ComplexF64` and `BigFloat`, at three sizes (about 4,000 cases). The inputs
are the matrix types of `LinearAlgebra` (`Matrix`, `Hermitian`, triangular,
`Diagonal`, `Bidiagonal`, `Tridiagonal`, `SymTridiagonal`, `SparseMatrixCSC`)
and those of OxiBLAS ([`CSRMatrix`](@ref), [`CSCMatrix`](@ref), [`COOMatrix`](@ref), [`BandedMatrix`](@ref),
[`PackedMatrix`](@ref)). Each case is compared with the matching reference:

| Family | Reference |
|---|---|
| products, BLAS 3 | `LinearAlgebra` (OpenBLAS, Accelerate on macOS; generic code for `Float16`/`BigFloat`) |
| dense factorizations, spectra, matrix functions | `LinearAlgebra` (LAPACK, or its generic code) |
| `\`, `OxiBLAS.ldiv`, `ldiv!` | the same problem with `LinearAlgebra` or SparseArrays |
| sparse factorizations | SparseArrays (UMFPACK, CHOLMOD, SPQR) |
| [`BandedMatrix`](@ref), [`PackedMatrix`](@ref) | the same matrix as `SparseMatrixCSC` or `Hermitian{Matrix}` |
| [`einsum`](@ref) | TensorOperations |

When `LinearAlgebra` has no method (for example `svd` of a `BigFloat` matrix,
or UMFPACK for `Float32`), only OxiBLAS is measured.

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark --threads=1 benchmark/run.jl                     # whole grid
julia --project=benchmark --threads=1 benchmark/run.jl lu 'ldiv!/*/f64/*'  # name prefixes and globs
julia --project=benchmark benchmark/run.jl --types=f32,f64 --sizes=s --families=solves
julia --project=benchmark benchmark/run.jl --allocs                        # allocation campaign only
OXIBLAS_REFERENCE=accelerate julia --project=benchmark benchmark/run.jl
```

**Method.**

- **Same conditions.** Same mathematical problem, same data and same number of
  threads for OxiBLAS and the reference.
- **Verification.** Each OxiBLAS result is checked by a backward error or a
  reconstruction, which does not need the reference.
- **Warm timings.** A first call compiles and warms up; the reported times are
  medians of warm samples (Chairmarks).
- **Allocations.** Allocation count and bytes of one warm call, on both sides.
  The report lists the in-place cases that allocate and the cases that
  allocate more than the reference.
- **Separate forms.** Allocating forms (`OxiBLAS.lu(A)`, `A \ b`), in-place
  forms (`ldiv!`, `mul!`, `lu!`) and sparse refactorizations
  (`lu!(F, A)`, `cholesky!(F, A)`) are distinct cases.
- **Targeted compilation.** Only the selected cases are compiled.
- **Outputs.** Status, median times, ratio, allocation counts and bytes, in
  CSV and Markdown under `benchmark/results/`, with summaries by family,
  operation, element type and input.

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
- `OxiBLAS.ldiv` and `\` on a [`CSRMatrix`](@ref);
- an [`einsum`](@ref) contraction.

The first call of these paths therefore no longer pays the compilation. Other
types and variants compile on first use.

Measurements on an Apple M5 Max, Julia 1.14, each path in a fresh process
(`benchmark/latency/run_all.jl`, results in `benchmark/latency/latency_*.tsv`):

| First call | Without workload | With workload |
|---|---|---|
| [`gemm!`](@ref) Float64 | 0.38 s | 0.09 s |
| `OxiBLAS.lu(A) \ b` | 2.85 s | 0.10 s |
| `OxiBLAS.cholesky` | 2.74 s | 0.01 s |
| `OxiBLAS.ldiv(A, b)` | 13.5 s | 0.13 s |
| sparse `A \ b` (Cholesky, LU) | 7.2 s | 0.005 s |
| [`einsum`](@ref) | 3.2 s | 0.06 s |
| `using OxiBLAS` | 0.03 s | 0.15 s |

**Cost of the workload.** Package precompilation goes from about 3 s to 20 s,
and its cache from 10 MB to about 50 MB.

**Disabling the workload.** Add `precompile_workload = false` to the
`[OxiBLAS]` section of `LocalPreferences.toml`.
