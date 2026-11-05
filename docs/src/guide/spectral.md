# Spectral and iterative algorithms

The dense spectral decompositions (Schur, Hermitian, SVD, QZ) are described in
[Dense factorizations](factorizations.md). This page covers the iterative and
partial methods, usable on dense or sparse matrices.

```@example spec
using OxiBLAS, SparseArrays, LinearAlgebra
n = 200
L = spdiagm(-1 => -ones(n-1), 0 => 2ones(n), 1 => -ones(n-1))   # 1-D Laplacian
r = lobpcg(CSRMatrix(L); nev=3, which=:smallest)
(r.converged, r.values)
```

## Eigenvalues

| Function | Method |
|---|---|
| `lanczos(A; nev, which)`, [`thick_restart_lanczos`](@ref) | Hermitian Lanczos, thick restart |
| [`arnoldi`](@ref), [`iram`](@ref), [`thick_restart_arnoldi`](@ref) | Arnoldi, implicit or thick restart |
| [`block_lanczos`](@ref), [`block_arnoldi`](@ref) (and `*_factor`) | block variants |
| `lobpcg(A; nev, B, which)` | LOBPCG, generalized problems possible |
| [`shift_invert`](@ref) | spectral transformation |
| [`polynomial_filter`](@ref), [`apply_filter!`](@ref), [`filter_response`](@ref) | Chebyshev filters ([`ChebyshevFilter`](@ref), [`FilterWorkspace`](@ref)) |
| [`eigenvalues_in_interval`](@ref), [`count_eigenvalues_in_interval`](@ref), [`polynomial_filtered_eigenvalues`](@ref), [`hermitian_spectral_bounds`](@ref) | interval search |

## Singular values

| Function | Method |
|---|---|
| [`sparse_svd_lanczos`](@ref), [`sparse_svd`](@ref) | partial sparse SVD (Golub–Kahan by default), [`SparseSVDWorkspace`](@ref) |
| `svd_selected(A; indices, interval)`, [`svd_truncated`](@ref), [`svd_randomized`](@ref) | partial dense SVDs |
| [`incremental_svd`](@ref), [`svd_add_rows!`](@ref), [`svd_add_columns!`](@ref), [`svd_snapshot`](@ref), [`svd_rank`](@ref) | SVD updated by rows or columns ([`IncrementalSVDFactor`](@ref)) |

## Stochastic estimators

[`trace_hutchinson`](@ref), [`trace_hutch_plus_plus`](@ref), [`trace_xtrace`](@ref) and
[`trace_xtrace_loo`](@ref) estimate the trace. [`stochastic_diagonal`](@ref),
[`stochastic_frobenius`](@ref) and [`stochastic_logdet`](@ref) estimate the diagonal, the
Frobenius norm and the log-determinant. All are configured by
`StochasticConfig(; num_probes, seed, ...)`, with an explicit seed and no
global state.
