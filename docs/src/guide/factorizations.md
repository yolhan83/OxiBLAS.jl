# Dense factorizations and LAPACK

The dense engines are written in Julia and reuse the native GEMM for their
blocked updates. Each one returns its own object ([`LUFactor`](@ref), [`QRFactor`](@ref)…),
usable with [`solve`](@ref) or through the [high-level API](highlevel.md), which wraps
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
| `lu_factor(A; blocksize, check, threaded, pivot=:partial)` | blocked LU; `pivot=:complete` or `:rook` ([`lu_complete`](@ref), [`lu_rook`](@ref)) |
| `cholesky_factor(A; blocksize, threaded)` | Cholesky, lower triangle |
| `ldlt_factor(A; uplo, hermitian, pivot=:bunchkaufman)`, [`bunchkaufman_factor`](@ref) | 1×1/2×2 D blocks, `A[p,p] ≈ L·D·Lᴴ`; `pivot=:none` possible |
| [`aasen_factor`](@ref) | Aasen factorization |
| `solve(F, b)`, [`solve_transposed`](@ref) | solve with a factorization |
| `least_squares(A, b)` | least squares, m ≥ n, full rank |
| `mixed_solve(A, b)` | LU in reduced precision, `Float64` refinement |
| [`expert_solve`](@ref), [`refine_solve`](@ref), [`expert_cholesky`](@ref), [`expert_symmetric`](@ref) | equilibration, iterative refinement, backward error |
| [`native_inverse`](@ref), [`cholesky_inverse`](@ref), [`pseudoinverse`](@ref) | inverses |
| [`determinant`](@ref), [`logabsdet_factor`](@ref), [`condition_number`](@ref), [`reciprocal_condition`](@ref), [`inverse_norm_estimate`](@ref) | determinant and conditioning |

## Orthogonalizations and rank

| Function | Result |
|---|---|
| `qr_factor(A; blocksize, threaded, pivot=false)`, [`qr_pivoted`](@ref) | compact WY QR; column pivoting `A[:,F.p] ≈ F.Q*F.R` |
| [`lq_factor`](@ref), [`ql_factor`](@ref), [`rq_factor`](@ref), [`complete_orthogonal`](@ref) | other orthogonal decompositions |
| [`apply_q!`](@ref), [`qr_q`](@ref) | apply or form Q |
| [`numerical_rank`](@ref), [`null_space`](@ref), [`left_null_space`](@ref), [`column_space`](@ref), [`row_space`](@ref) | rank and subspaces |

## Eigenvalues and singular values

| Function | Result |
|---|---|
| [`hessenberg_factor`](@ref), [`hessenberg_q`](@ref), [`apply_hessenberg_q!`](@ref) | Hessenberg reduction |
| [`schur_factor`](@ref), [`schur_multishift`](@ref), [`eigenvalues`](@ref), [`eigen_general`](@ref) | Schur (shifted QR), general eigenvalues and eigenvectors |
| [`schur_reorder`](@ref), [`schur_eigenvectors`](@ref), [`trevc_right`](@ref), [`trevc_left`](@ref), [`trsna_s`](@ref), [`trsna_sep`](@ref), [`schur_condition`](@ref) | reordering, vectors, conditioning |
| [`tridiagonal_factor`](@ref), [`eigen_tridiagonal`](@ref), [`eigen_tridiagonal_dc`](@ref), [`eigen_tridiagonal_mrrr`](@ref) | Hermitian tridiagonal (implicit QR, divide and conquer, MRRR) |
| [`eigen_hermitian`](@ref), [`eigen_hermitian_dc`](@ref), [`eigen_hermitian_mrrr`](@ref), [`eigen_generalized`](@ref) | standard and generalized Hermitian problems |
| [`qz_factor`](@ref), [`eigen_pencil`](@ref), [`qz_reorder`](@ref), [`qz_condition`](@ref), [`generalized_sylvester`](@ref) | matrix pencils (real or complex QZ) |
| `svd_factor(A; algorithm)`, [`singular_values`](@ref) | SVD: one-sided Jacobi (default) or `:golub_kahan` |
| [`svd_bidiagonal`](@ref), [`svd_divide_conquer`](@ref), [`svd_truncated`](@ref), [`svd_selected`](@ref), [`svd_randomized`](@ref) | variants and partial SVDs |
| [`bidiagonal_factor`](@ref), [`balance_matrix`](@ref), [`balance_backtransform`](@ref) | auxiliary reductions |

**SVD.** The default Jacobi guarantees relative accuracy on small singular
values, at several times the cost of `gesdd`. `algorithm=:golub_kahan` is the
fast path (see the high-level suite in [Performance](performance.md)).

## Matrix functions

[`matrix_exp`](@ref), [`matrix_log`](@ref), [`matrix_sqrt`](@ref), [`matrix_power`](@ref), [`matrix_sign`](@ref),
[`matrix_sin`](@ref), [`matrix_cos`](@ref), the Fréchet derivatives ([`frechet_exp`](@ref),
[`frechet_log`](@ref), [`frechet_sqrt`](@ref), [`frechet_matrix`](@ref)), [`matrix_norm`](@ref),
[`kron_product`](@ref), [`khatri_rao`](@ref), [`kron_sum`](@ref) and [`kronecker_action`](@ref). The
high-level versions `OxiBLAS.exp`, `log`, `sqrt`, `sin` and `cos` keep the
structure when it is guaranteed (see the [high-level API](highlevel.md)).

## Extended precision

The generic algorithms accept other floating-point types (`BigFloat`,
`Float16`, extended-precision types provided by other packages), as do
[`gemm_extended!`](@ref) and [`dot_extended`](@ref).

## Errors

`SingularException` and `PosDefException` are those of `LinearAlgebra`
(re-exported under the same names). [`ConvergenceError`](@ref) reports that an
iteration limit was reached.
