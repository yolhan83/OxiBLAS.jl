# SPDX-License-Identifier: Apache-2.0
# Docstrings of the dense LAPACK family: factorization types, solves, matrix
# functions, ranks and subspaces. Attached to the bindings.

# ---- factorization types -------------------------------------------------------

@doc raw"""
    LUFactor{T}

Result of [`lu_factor`](@ref): `P·A = L·U` with partial pivoting.

- `factors`: `L` (unit lower, below the diagonal) and `U` (upper) packed in one
  matrix; `F.L` and `F.U` extract them.
- `pivots`: LAPACK interchange sequence (row `j` was swapped with
  `pivots[j]`); `p`: the resulting permutation, `A[p,:] == L*U`.
- `info`: 0, or the index of the first zero pivot.

Solve with [`solve`](@ref)`(F, b)`; the high-level `OxiBLAS.lu` wraps it in a
`LinearAlgebra.Factorization`.
""" LUFactor

@doc raw"""
    CompleteLUFactor{T}

Result of [`lu_complete`](@ref) or [`lu_rook`](@ref): `A[p,q] = L·U` with row
and column pivoting. Fields `factors`, `p`, `q`, `info`, `pivot` (`:complete`
or `:rook`).
""" CompleteLUFactor

@doc raw"""
    CholeskyFactor{T}

Result of [`cholesky_factor`](@ref): `A = L·Lᴴ`, `L` lower triangular (field
`L`). Solve with [`solve`](@ref).
""" CholeskyFactor

@doc raw"""
    LDLFactor{T}

Result of [`ldlt_factor`](@ref) / [`bunchkaufman_factor`](@ref):
`A[p,p] = L·D·Lᴴ` (or `L·D·Lᵀ` when `hermitian` is false), `L` unit lower,
`D` block diagonal with 1×1 and 2×2 blocks (`blocks` lists their sizes).
""" LDLFactor

@doc raw"""
    QRFactor{T}

Result of [`qr_factor`](@ref): Householder QR in compact form (`factors`
holds `R` and the reflectors, `tau` their scalars). `F.Q`/`F.R` extract the
factors; [`qr_q`](@ref) forms `Q` explicitly; [`solve`](@ref) gives the
least-squares solution.
""" QRFactor

@doc raw"""
    PivotedQRFactor{T}

Result of [`qr_pivoted`](@ref): `A[:,p] = Q·R` with column pivoting (fields
`qr::QRFactor` and `p`). Rank-revealing; [`solve`](@ref) gives the
least-squares solution.
""" PivotedQRFactor

@doc raw"""
    OrthogonalFactor{T}

Result of the RQ, QL and LQ factorizations: fields `triangular`, `orthogonal`
and `kind` (`:RQ`, `:QL` or `:LQ`).
""" OrthogonalFactor

@doc raw"""
    CompleteOrthogonalFactor{T}

Result of [`complete_orthogonal`](@ref): `A[:,p] = Q·[T 0; 0 0]·Zᴴ` with the
numerical `rank`. [`solve`](@ref) gives the minimum-norm least-squares
solution.
""" CompleteOrthogonalFactor

@doc raw"""
    SVDFactor{T,R}

Result of [`svd_factor`](@ref) and the other SVD routines: `A ≈ U·Diagonal(S)·Vt`.
`U` and `Vt` are `nothing` when the vectors were not requested; `S` is sorted
in decreasing order; `sweeps` counts the iterations of the algorithm.
""" SVDFactor

@doc raw"""
    EigenFactor{T,R}

Result of the Hermitian eigensolvers ([`eigen_hermitian`](@ref) and variants):
real `values` in increasing order, `vectors` (columns, or `nothing`),
`iterations`.
""" EigenFactor

@doc raw"""
    SchurFactor{T,C}

Result of [`schur_factor`](@ref): `A = Z·T·Zᴴ`, `T` quasi-upper triangular
(real Schur form, 2×2 blocks for complex pairs) or upper triangular (complex),
`values` the eigenvalues, `iterations` the QR sweeps.
""" SchurFactor

@doc raw"""
    QZFactor{T,C}

Result of [`qz_factor`](@ref): generalized Schur form `A = Q·S·Zᴴ`,
`B = Q·T·Zᴴ`. The generalized eigenvalues are `alpha ./ beta` (`F.values`).
""" QZFactor

@doc raw"""
    HessenbergFactor{T,R}

Result of [`hessenberg_factor`](@ref): `A = Q·H·Qᴴ` with `H` upper Hessenberg,
stored as reflectors (`factors`, `tau`). [`hessenberg_q`](@ref) forms `Q`.
""" HessenbergFactor

@doc raw"""
    TridiagonalFactor{T,R}

Result of [`tridiagonal_factor`](@ref): Hermitian `A = Q·Tri·Qᴴ` with `Tri`
real symmetric tridiagonal (`diag`, `offdiag`). [`tridiagonal_q`](@ref) forms
`Q`.
""" TridiagonalFactor

@doc raw"""
    BalanceFactor{T,R}

Result of [`balance_matrix`](@ref): `matrix = Diagonal(1 ./ scales)·A[p,p]·Diagonal(scales)`
(fields `matrix`, `permutation`, `scales`, `ilo`, `ihi`). Eigenvectors of the
balanced matrix go back with [`balance_backtransform`](@ref).
""" BalanceFactor

@doc raw"""
    SchurWorkspace(T, n; blocksize=64, nshifts=6)

Reusable buffers of the multishift QR sweeps of [`schur_factor`](@ref) for
matrices of order up to `n`. Pass it as `workspace=` to avoid allocating them
at every call. Not shareable between concurrent calls.
""" SchurWorkspace

@doc raw"""
    ConvergenceError(algorithm, iterations, active)

Exception raised when an iterative algorithm (QR sweeps, Jacobi, MRRR…) does
not converge within its iteration budget: `algorithm` names it, `active` is the
index range still unconverged.
""" ConvergenceError

# ---- factorizations and solves -----------------------------------------------

@doc raw"""
    solve(F, b)

Solves `A·x = b` with a factorization `F` of `A` (`LUFactor`, `CholeskyFactor`,
`LDLFactor`, `QRFactor`, `PivotedQRFactor`, `CompleteOrthogonalFactor`,
`AasenFactor`, band and packed factors, sparse factors…). `b` is a vector or a
matrix (several right-hand sides) and is not modified. For QR factors the
result is the least-squares solution.

```julia
F = lu_factor(A); x = solve(F, b)
```
""" solve

@doc raw"""
    solve_transposed(F, b; trans=:C)

Solves `Aᴴ·x = b` (`trans=:C`) or `Aᵀ·x = b` (`trans=:T`) with an `LUFactor` or
`CompleteLUFactor` of `A`, without refactorizing.
""" solve_transposed

@doc raw"""
    lu_complete(A; check=true, threaded=true) -> CompleteLUFactor

LU with complete pivoting (`A[p,q] = L·U`): the most stable LU, at the cost of
an `O(n²)` pivot search per step. Same as `lu_factor(A; pivot=:complete)`.
""" lu_complete

@doc raw"""
    lu_rook(A; check=true, threaded=true) -> CompleteLUFactor

LU with rook pivoting (a pivot largest in both its row and its column):
nearly as stable as complete pivoting, cheaper on average. Same as
`lu_factor(A; pivot=:rook)`.
""" lu_rook

@doc raw"""
    bunchkaufman_factor(A; uplo=:L, hermitian=true) -> LDLFactor

Symmetric indefinite factorization with Bunch–Kaufman pivoting (1×1 and 2×2
pivots). Same as `ldlt_factor(A; pivot=:bunchkaufman)`.
""" bunchkaufman_factor

@doc raw"""
    ldlt_factor(A; uplo=:L, hermitian=true, pivot=:bunchkaufman, threaded=true, blocksize=nothing) -> LDLFactor

Factorization `A[p,p] = L·D·Lᴴ` (or `L·D·Lᵀ` when `hermitian` is false), where `L` is unit lower
triangular and `D` is block diagonal with 1×1 and 2×2 blocks (`blocks` lists their sizes).

- `pivot=:bunchkaufman` selects the bounded Bunch–Kaufman pivoting strategy.
- `pivot=:none` disables pivoting (suitable for positive definite matrices).
- `uplo=:L` or `:U`: only the referenced triangle of `A` is accessed.
- Uses blocked panel updates and multithreaded GEMMT trailing matrix rank-k updates.
""" ldlt_factor

@doc raw"""
    bidiagonal_factor(A) -> BidiagonalFactor

Householder reduction `A = Q·B·Pᴴ` with `B` upper bidiagonal (lower for a wide
`A`, which is reduced through its adjoint): the first stage of the
Golub–Kahan SVD.
""" bidiagonal_factor

@doc raw"""
    band_cholesky(A::BandedMatrix) -> BandCholeskyFactor

Cholesky factorization of a Hermitian positive definite banded matrix, in band
storage (no fill outside the band). Solve with [`solve`](@ref).
""" band_cholesky

@doc raw"""
    expert_cholesky(A, b; uplo=:L, maxiter=8, rtol=8eps) -> NamedTuple

Expert driver: Cholesky of the Hermitian `A` (only the `uplo` triangle is read)
then iterative refinement (see [`refine_solve`](@ref)). Returns
`(; x, converged, iterations, backward_error, residual_norm, rcond_estimate,
forward_error_estimate)`.
""" expert_cholesky

@doc raw"""
    expert_symmetric(A, b; uplo=:L, hermitian=true, maxiter=8, rtol=8eps) -> NamedTuple

Expert driver for symmetric or Hermitian indefinite systems: Bunch–Kaufman
LDLᵀ, then iterative refinement. Same result fields as
[`expert_cholesky`](@ref).
""" expert_symmetric

@doc raw"""
    backward_error(A, x, b) -> (; error, residual)

Componentwise backward error of an approximate solution `x` of `A·x = b`:
`max |b - A·x|ᵢ / (|A|·|x| + |b|)ᵢ`, and the residual `b - A·x`.
""" backward_error

@doc raw"""
    equilibrate(A; radix=false) -> (; matrix, row_scale, column_scale)

Row then column scaling so that the largest entry of every row and column has
magnitude 1: `matrix = Diagonal(1 ./ row_scale)·A·Diagonal(1 ./ column_scale)`.
`radix=true` rounds the scales to powers of 2 (scaling without rounding error).
""" equilibrate

@doc raw"""
    reciprocal_condition(A; maxiter=8)

Estimate of `1/κ₁(A)` (LAPACK `gecon` style: Hager–Higham estimate of
`‖A⁻¹‖₁` from an LU factorization). 0 for a singular `A`. An estimate, not a
bound.
""" reciprocal_condition

@doc raw"""
    condition_number(A; norm=:two)

Condition number `‖A‖·‖A⁻¹‖`. `norm=:two` uses the singular values
(`σmax/σmin`, `Inf` when singular, also for rectangular `A`); `:one`, `:inf`
and `:fro` form the inverse explicitly.
""" condition_number

@doc raw"""
    determinant(A)

Determinant from an LU factorization (0 when a pivot is exactly zero). For
large matrices prefer [`logabsdet_factor`](@ref), which does not overflow.
""" determinant

@doc raw"""
    logabsdet_factor(A) -> (logabs, phase)
    logabsdet_factor(F::Union{MultifrontalLUFactor, SparseStaticLUFactor, SparseLDLFactor})

`log|det(A)|` and the phase `det(A)/|det(A)|` (the sign for real matrices),
from a dense LU or from an existing sparse factorization.
""" logabsdet_factor

@doc raw"""
    native_inverse(A; threaded=true) -> Matrix

Explicit inverse through LU (LAPACK `getri` scheme: inverse of `U`, then a
triangular solve with `L`, then the permutation). Raises `SingularException`
for a singular `A`. Solving with a factorization is cheaper and more accurate
when only `A⁻¹·b` is needed.
""" native_inverse

@doc raw"""
    pseudoinverse(A; rtol=eps·max(m,n), atol=0, algorithm=:golub_kahan) -> Matrix

Moore–Penrose pseudoinverse from the SVD; singular values below
`max(atol, rtol·σmax)` are treated as zero.
""" pseudoinverse

@doc raw"""
    numerical_rank(A; rtol=eps·max(m,n), atol=0) -> Int

Number of singular values above `max(atol, rtol·σmax)`.
""" numerical_rank

@doc raw"""
    null_space(A; rtol, atol) -> Matrix

Orthonormal basis of the null space `{x : A·x = 0}` (right singular vectors of
the singular values below the tolerance, as for [`numerical_rank`](@ref)).
""" null_space

@doc raw"""
    left_null_space(A; rtol, atol) -> Matrix

Orthonormal basis of `{y : Aᴴ·y = 0}`, i.e. `null_space(adjoint(A))`.
""" left_null_space

@doc raw"""
    column_space(A; rtol, atol) -> Matrix

Orthonormal basis of the range of `A` (left singular vectors of the singular
values above the tolerance).
""" column_space

@doc raw"""
    row_space(A; rtol, atol) -> Matrix

Orthonormal basis of the range of `Aᴴ`, i.e. `column_space(adjoint(A))`.
""" row_space

@doc raw"""
    matrix_norm(A, kind=:fro)

Matrix norms: `:fro` (Frobenius), `:one` (max column sum), `:inf` (max row
sum), `:max` (largest entry), `:two` (largest singular value) and `:nuclear`
(sum of the singular values).
""" matrix_norm

# ---- spectra ---------------------------------------------------------------------

@doc raw"""
    eigenvalues(A; kwargs...) -> Vector

Eigenvalues of a general square matrix, from the Schur form
([`schur_factor`](@ref)); complex for a real `A` with complex pairs.
""" eigenvalues

@doc raw"""
    eigen_hermitian_dc(A; uplo=:U, vectors=true, kwargs...) -> EigenFactor

Hermitian eigendecomposition by tridiagonal reduction and Cuppen
divide-and-conquer: fastest for all eigenvectors of large matrices.
""" eigen_hermitian_dc

@doc raw"""
    eigen_hermitian_mrrr(A; uplo=:U, vectors=true, indices=nothing, interval=nothing, kwargs...) -> EigenFactor

Hermitian eigendecomposition by tridiagonal reduction and MRRR (multiple
relatively robust representations): `O(n²)` work for the eigenvectors, and a
subset by `indices` (e.g. `1:10`) or by an `interval` `(lo, hi]`.
""" eigen_hermitian_mrrr

@doc raw"""
    eigen_pencil(A, B; vectors=true, left=false, kwargs...) -> NamedTuple

Generalized eigenproblem `A·x = λ·B·x` for a general pencil through the QZ
algorithm. Returns `(; values, alpha, beta, vectors, left_vectors, qz)`, with
`values = alpha ./ beta` (infinite when `beta` is zero). For Hermitian `A` and
positive definite `B` use [`eigen_generalized`](@ref).
""" eigen_pencil

@doc raw"""
    schur_eigenvectors(F::SchurFactor; left=false) -> Matrix

Eigenvectors of the original matrix from its Schur factorization (back
substitution on `T`, then multiplication by `Z`); `left=true` gives the left
eigenvectors.
""" schur_eigenvectors

@doc raw"""
    schur_condition(F::SchurFactor; split=nothing) -> (; eigenvalue, spectral_gap, subspace)

Condition estimates of a Schur factorization (LAPACK `trsna`/`trsen`):
reciprocal condition numbers of the eigenvalues, separations of the
eigenvalues, and of the invariant subspace given by `split`.
""" schur_condition

@doc raw"""
    trevc_left(S; packed=T<:Real) -> Matrix

Left eigenvectors of a matrix in Schur form `S` (LAPACK `trevc`), the
counterpart of [`trevc_right`](@ref).
""" trevc_left

@doc raw"""
    hessenberg_q(F::HessenbergFactor; algorithm=:auto, blocksize=32) -> Matrix

Forms the unitary `Q` of a Hessenberg reduction explicitly (blocked
application of the reflectors).
""" hessenberg_q

@doc raw"""
    tridiagonal_q(F::TridiagonalFactor; algorithm=:auto, blocksize=32) -> Matrix

Forms the unitary `Q` of a tridiagonal reduction explicitly, including the
phases that make the off-diagonal real.
""" tridiagonal_q

@doc raw"""
    balance_backtransform(B::BalanceFactor, V; left=false) -> Matrix

Maps eigenvectors `V` of the balanced matrix `B.matrix` back to eigenvectors
of the original matrix (LAPACK `gebak`). `left=true` for left eigenvectors.
""" balance_backtransform

@doc raw"""
    singular_values(A; kwargs...) -> Vector

Singular values only, in decreasing order (no vectors computed); keywords as
[`svd_factor`](@ref) (`algorithm=:jacobi` by default, `:golub_kahan`,
`:divide_conquer`).
""" singular_values

@doc raw"""
    svd_bidiagonal(A; algorithm=:golub_kahan, vectors=true, full=false, maxiter=..., rtol=..., leafsize=64, threaded=true) -> SVDFactor

Singular value decomposition of a matrix through bidiagonal reduction:
`A = U·Diagonal(S)·Vt`.

- `algorithm=:golub_kahan`: implicitly shifted Golub–Kahan QR with SIMD Givens rotations
  applied in L1/L2 cache-resident tiles across threads.
- `algorithm=:divide_conquer`: divide-and-conquer secular equation solver (LAPACK `gesdd` scheme).
- `vectors=false`: computes singular values only (same as [`singular_values`](@ref)).
- `full=true`: square unitary factors `U` and `Vt` completing the null spaces.
""" svd_bidiagonal

@doc raw"""
    svd_divide_conquer(A; vectors=true, full=false, kwargs...) -> SVDFactor

SVD by bidiagonal reduction and divide-and-conquer (LAPACK `gesdd` scheme):
the fastest full SVD with vectors on large matrices. Same as
`svd_bidiagonal(A; algorithm=:divide_conquer)`.
""" svd_divide_conquer

# ---- matrix functions and Kronecker products ----------------------------------

@doc raw"""
    matrix_power(A, p::Integer; threaded=true)
    matrix_power(A, p::Number; threaded=true)

`A^p`. Integer `p`: repeated squaring (`p < 0` through the inverse). Other `p`:
`exp(p·log(A))` (principal branch).
""" matrix_power

@doc raw"""
    matrix_sin(A; kwargs...) -> Matrix

Matrix sine, `(exp(iA) - exp(-iA))/(2i)`; complex result for a real `A`
(the imaginary part is rounding noise).
""" matrix_sin

@doc raw"""
    matrix_cos(A; kwargs...) -> Matrix

Matrix cosine, `(exp(iA) + exp(-iA))/2`; complex result for a real `A`.
""" matrix_cos

@doc raw"""
    frechet_exp(A, E; kwargs...) -> (; value, derivative)

`exp(A)` and its Fréchet derivative in the direction `E`, from the upper
blocks of `exp([A E; 0 A])`.
""" frechet_exp

@doc raw"""
    frechet_log(A, E; kwargs...) -> (; value, derivative)

`log(A)` and its Fréchet derivative in the direction `E`.
""" frechet_log

@doc raw"""
    frechet_sqrt(A, E; kwargs...) -> (; value, derivative)

`sqrt(A)` and its Fréchet derivative in the direction `E`.
""" frechet_sqrt

@doc raw"""
    kron_product(A, B) -> Matrix

Kronecker product `A ⊗ B`.
""" kron_product

@doc raw"""
    kron_sum(A, B) -> Matrix

Kronecker sum `A ⊗ I + I ⊗ B` of two square matrices.
""" kron_sum

@doc raw"""
    kronecker_action(A, B, x) -> Vector

`(A ⊗ B)·x` without forming the Kronecker product: `vec(B·X·Aᵀ)` with `X`
the reshape of `x`. Cost `O(mn(p+q))` instead of `O(mnpq)`.
""" kronecker_action

@doc raw"""
    khatri_rao(A, B) -> Matrix

Column-wise Kronecker (Khatri–Rao) product: column `j` is `A[:,j] ⊗ B[:,j]`.
""" khatri_rao
