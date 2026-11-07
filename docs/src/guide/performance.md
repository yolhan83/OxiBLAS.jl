# Performance and Benchmarks

OxiBLAS provides native, highly optimized linear algebra kernels and algorithms across all numeric types (`Float16`, `Float32`, `Float64`, `ComplexF32`, `ComplexF64`, and `BigFloat`) and matrix representations (dense, banded, packed, CSR, CSC, and COO sparse formats).

This page provides an interactive performance dashboard tracking verified benchmarks comparing OxiBLAS with standard references (`LinearAlgebra` with OpenBLAS / Apple Accelerate, `SparseArrays` SuiteSparse, and `TensorOperations`), followed by detailed benchmark methodology and reproduction instructions.

---

## Interactive Benchmark Explorer

Explore the complete high-level benchmark suite. You can filter by domain family, operation, matrix input type, numeric precision, and problem size ($s$, $m$, $l$). Toggle between **1 Thread** and **17 Threads** to evaluate parallel scaling.

```@raw html
<div id="oxiblas-dashboard-app" data-results-src="../assets/benchmarks/results.json"></div>
```

---

## Suite Architecture & Methodology

The automated benchmark harness (`benchmark/run.jl`) measures an exhaustive grid of operations and representations against standard references:

| Family | Operation Scope | Reference Library |
| :--- | :--- | :--- |
| **Products & BLAS 3** | `mul`, `mul!`, GEMM, SpMM, SpGEMM | `LinearAlgebra` (OpenBLAS / Accelerate; generic fallback for `Float16`/`BigFloat`) |
| **Factorizations** | LU, Cholesky, QR, LQ, SVD | `LinearAlgebra` (LAPACK; generic fallback for unsupported precisions) |
| **Solves & Inversion** | `\`, `ldiv`, `ldiv!` | `LinearAlgebra` or `SparseArrays` (UMFPACK / CHOLMOD / SPQR) |
| **Structured Formats** | [`BandedMatrix`](@ref), [`PackedMatrix`](@ref) | Equivalent problem formulated as `SparseMatrixCSC` or `Hermitian{Matrix}` |
| **Tensors** | [`einsum`](@ref) contractions | `TensorOperations` |

When standard libraries do not implement a method (for example, Cholesky or SVD on `BigFloat` matrices, or UMFPACK sparse factorizations on `Float32` / `ComplexF32`), only OxiBLAS is measured and validated.

### Rigorous Evaluation Criteria

1. **Identical Conditions:** Every case executes on identical random matrices with reproducible seeds and identical thread allocation (`Threads.nthreads()`).
2. **Mathematical Verification:** Every OxiBLAS computation is checked via backward error or direct reconstruction (e.g. $\|A - LU\| / \|A\|$), ensuring correctness independent of reference availability.
3. **Warm-up & Timing:** A preliminary call compiles and warms the CPU cache; reported timings represent the median of warm execution samples measured with `Chairmarks.jl`.
4. **Allocation Profiling:** Memory allocations (count and total bytes) are measured using `Base.@allocations` and `Base.@allocated`. In-place routines (`mul!`, `ldiv!`, `lu!`) are audited for zero-allocation performance.
5. **Distinct Execution Forms:** Allocating calls (`OxiBLAS.lu(A)`, `A \ b`), in-place mutations (`ldiv!(y, F, b)`, `mul!(C, A, B)`), and sparse refactorizations (`lu!(F, A)`) are benchmarked as distinct cases.

---

## Reproducing Benchmarks Locally

Benchmark campaigns can be executed via the Antigravity / Julia command line from the repository root:

```sh
# Instantiate benchmark dependencies
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'

# Run the complete suite on 1 thread
julia --project=benchmark --threads=1 benchmark/run.jl

# Run specific families or operations with custom patterns
julia --project=benchmark --threads=17 benchmark/run.jl --families=products,factorizations
julia --project=benchmark benchmark/run.jl --types=f32,f64 --sizes=s,m --families=solves

# Run an allocation-only profiling audit (no timing overhead)
julia --project=benchmark benchmark/run.jl --allocs

# Publish benchmark results into the documentation snapshot
julia --project=benchmark benchmark/publish_results.jl
```

---

## Latency and Precompilation

OxiBLAS integrates workload precompilation using `PrecompileTools` to eliminate first-call compilation latency for standard workflows:

- `Float32` and `Float64` dense GEMM;
- Dense LU and Cholesky factorizations;
- Sparse factorizations and iterative triangular solves;
- Sparse matrix-vector multiplications on [`CSRMatrix`](@ref) and [`CSCMatrix`](@ref);
- Tensor [`einsum`](@ref) contractions.

The table below measures first-call latency in a fresh Julia process on Apple Silicon:

| First Call Workflow | Without Precompile Workload | With Precompile Workload | Speedup |
| :--- | :---: | :---: | :---: |
| [`gemm!`](@ref) Float64 | 0.38 s | **0.09 s** | 4.2× |
| `OxiBLAS.lu(A) \ b` | 2.85 s | **0.10 s** | 28.5× |
| `OxiBLAS.cholesky(A)` | 2.74 s | **0.01 s** | 274× |
| `OxiBLAS.ldiv(A, b)` | 13.5 s | **0.13 s** | 103× |
| Sparse `A \ b` (Cholesky, LU) | 7.2 s | **0.005 s** | 1,440× |
| [`einsum`](@ref) | 3.2 s | **0.06 s** | 53× |
| `using OxiBLAS` overhead | 0.03 s | **0.15 s** | — |

To disable precompilation workloads in resource-constrained container builds, add `precompile_workload = false` to the `[OxiBLAS]` section in `LocalPreferences.toml`.
