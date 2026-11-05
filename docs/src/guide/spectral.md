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
| `lanczos(A; nev, which)`, `thick_restart_lanczos` | Hermitian Lanczos, thick restart |
| `arnoldi`, `iram`, `thick_restart_arnoldi` | Arnoldi, implicit or thick restart |
| `block_lanczos`, `block_arnoldi` (and `*_factor`) | block variants |
| `lobpcg(A; nev, B, which)` | LOBPCG, generalized problems possible |
| `shift_invert` | spectral transformation |
| `polynomial_filter`, `apply_filter!`, `filter_response` | Chebyshev filters (`ChebyshevFilter`, `FilterWorkspace`) |
| `eigenvalues_in_interval`, `count_eigenvalues_in_interval`, `polynomial_filtered_eigenvalues`, `hermitian_spectral_bounds` | interval search |

## Singular values

| Function | Method |
|---|---|
| `sparse_svd_lanczos`, `sparse_svd` | partial sparse SVD (Golub–Kahan by default), `SparseSVDWorkspace` |
| `svd_selected(A; indices, interval)`, `svd_truncated`, `svd_randomized` | partial dense SVDs |
| `incremental_svd`, `svd_add_rows!`, `svd_add_columns!`, `svd_snapshot`, `svd_rank` | SVD updated by rows or columns (`IncrementalSVDFactor`) |

## Stochastic estimators

`trace_hutchinson`, `trace_hutch_plus_plus`, `trace_xtrace` and
`trace_xtrace_loo` estimate the trace. `stochastic_diagonal`,
`stochastic_frobenius` and `stochastic_logdet` estimate the diagonal, the
Frobenius norm and the log-determinant. All are configured by
`StochasticConfig(; num_probes, seed, ...)`, with an explicit seed and no
global state.
