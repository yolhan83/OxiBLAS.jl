# Dense factorizations and LAPACK

The dense engines are written in Julia and reuse the native GEMM for their
blocked updates. Each one returns its own object (`LUFactor`, `QRFactor`…),
usable with `solve` or through the [high-level API](highlevel.md), which wraps
it in a `LinearAlgebra.Factorization`.

```@example fact
using OxiBLAS, LinearAlgebra
A = randn(80, 80) + 80I; b = randn(80)
F = lu_factor(A)                 # F.L*F.U ≈ A[F.p,:]
x = solve(F, b)
H = Hermitian(A * A')
C = cholesky_factor(Matrix(H))   # lower triangle, F.L*F.L' ≈ A
Q = qr_factor(randn(100, 40))    # least squares: least_squares(Q, y)
norm(A * x - b) / norm(b)
```

## Linear systems

| Function | Result |
|---|---|
| `lu_factor(A; blocksize, check, threaded, pivot=:partial)` | blocked LU; `pivot=:complete` or `:rook` (`lu_complete`, `lu_rook`) |
| `cholesky_factor(A; blocksize, threaded)` | Cholesky, lower triangle |
| `ldlt_factor(A; uplo, hermitian, pivot=:bunchkaufman)`, `bunchkaufman_factor` | 1×1/2×2 D blocks, `A[p,p] ≈ L·D·Lᴴ`; `pivot=:none` possible |
| `aasen_factor` | Aasen factorization |
| `solve(F, b)`, `solve_transposed` | solve with a factorization |
| `least_squares(A, b)` | least squares, m ≥ n, full rank |
| `mixed_solve(A, b)` | LU in reduced precision, `Float64` refinement |
| `expert_solve`, `refine_solve`, `expert_cholesky`, `expert_symmetric` | equilibration, iterative refinement, backward error |
| `native_inverse`, `cholesky_inverse`, `pseudoinverse` | inverses |
| `determinant`, `logabsdet_factor`, `condition_number`, `reciprocal_condition`, `inverse_norm_estimate` | determinant and conditioning |

## Orthogonalizations and rank

| Function | Result |
|---|---|
| `qr_factor(A; blocksize, threaded, pivot=false)`, `qr_pivoted` | compact WY QR; column pivoting `A[:,F.p] ≈ F.Q*F.R` |
| `lq_factor`, `ql_factor`, `rq_factor`, `complete_orthogonal` | other orthogonal decompositions |
| `apply_q!`, `qr_q` | apply or form Q |
| `numerical_rank`, `null_space`, `left_null_space`, `column_space`, `row_space` | rank and subspaces |

## Eigenvalues and singular values

| Function | Result |
|---|---|
| `hessenberg_factor`, `hessenberg_q`, `apply_hessenberg_q!` | Hessenberg reduction |
| `schur_factor`, `schur_multishift`, `eigenvalues`, `eigen_general` | Schur (shifted QR), general eigenvalues and eigenvectors |
| `schur_reorder`, `schur_eigenvectors`, `trevc_right`, `trevc_left`, `trsna_s`, `trsna_sep`, `schur_condition` | reordering, vectors, conditioning |
| `tridiagonal_factor`, `eigen_tridiagonal`, `eigen_tridiagonal_dc`, `eigen_tridiagonal_mrrr` | Hermitian tridiagonal (implicit QR, divide and conquer, MRRR) |
| `eigen_hermitian`, `eigen_hermitian_dc`, `eigen_hermitian_mrrr`, `eigen_generalized` | standard and generalized Hermitian problems |
| `qz_factor`, `eigen_pencil`, `qz_reorder`, `qz_condition`, `generalized_sylvester` | matrix pencils (real or complex QZ) |
| `svd_factor(A; algorithm)`, `singular_values` | SVD: one-sided Jacobi (default) or `:golub_kahan` |
| `svd_bidiagonal`, `svd_divide_conquer`, `svd_truncated`, `svd_selected`, `svd_randomized` | variants and partial SVDs |
| `bidiagonal_factor`, `balance_matrix`, `balance_backtransform` | auxiliary reductions |

**SVD.** The default Jacobi guarantees relative accuracy on small singular
values, at several times the cost of `gesdd`. `algorithm=:golub_kahan` is the
fast path (see the high-level suite in [Performance](performance.md)).

## Matrix functions

`matrix_exp`, `matrix_log`, `matrix_sqrt`, `matrix_power`, `matrix_sign`,
`matrix_sin`, `matrix_cos`, the Fréchet derivatives (`frechet_exp`,
`frechet_log`, `frechet_sqrt`, `frechet_matrix`), `matrix_norm`,
`kron_product`, `khatri_rao`, `kron_sum` and `kronecker_action`. The
high-level versions `OxiBLAS.exp`, `log`, `sqrt`, `sin` and `cos` keep the
structure when it is guaranteed (see the [high-level API](highlevel.md)).

## Extended precision

The generic algorithms accept other floating-point types (`BigFloat`,
`Float16`, extended-precision types provided by other packages), as do
`gemm_extended!` and `dot_extended`.

## Errors

`SingularException` and `PosDefException` are those of `LinearAlgebra`
(re-exported under the same names). `ConvergenceError` reports that an
iteration limit was reached.
