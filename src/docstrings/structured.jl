# SPDX-License-Identifier: Apache-2.0
# Docstrings of the structured family (packed and banded storage, their BLAS
# and factorizations) and of the remaining families: tensors, spectral,
# graphs, utils.

# ---- packed and banded storage -----------------------------------------------------

@doc raw"""
    PackedMatrix(data, n; uplo=:U, kind=:symmetric, unit=false)

BLAS packed storage of a symmetric, Hermitian or triangular matrix of order
`n`: the `uplo` triangle stored column by column in `data` (length
`n(n+1)/2`). `kind` is `:symmetric`, `:hermitian` or `:triangular` (`unit`:
implicit unit diagonal). Built from a dense matrix with
[`pack_matrix`](@ref); used by [`spmv!`](@ref), [`tpsv!`](@ref),
[`packed_cholesky`](@ref) and the high-level API (`OxiBLAS.cholesky`, `\`).

```julia
P = pack_matrix(H; uplo=:U, kind=:symmetric)
y = zeros(n); spmv!(y, P, x)          # y = H*x
```
""" PackedMatrix

@doc raw"""
    pack_matrix(A; uplo=:U, kind=:symmetric, unit=false) -> PackedMatrix

Copies the `uplo` triangle of `A` into a [`PackedMatrix`](@ref).
""" pack_matrix

@doc raw"""
    band_matrix(A; kl=0, ku=0, kind=:general, uplo=:U, unit=false) -> BandedMatrix

Copies the band of `A` into a [`BandedMatrix`](@ref): `kl` subdiagonals and
`ku` superdiagonals for `kind=:general`; for `:symmetric`, `:hermitian` and
`:triangular` the `uplo` triangle with `max(kl, ku)` off-diagonals. Only the
band is read.
""" band_matrix

@doc raw"""
    spmv!(y, P::PackedMatrix, x, α=1, β=0; kernel=:auto, threaded=true) -> y

Symmetric packed matrix–vector product `y ← α·P·x + β·y` (BLAS `spmv`).
""" spmv!

@doc raw"""
    hpmv!(y, P::PackedMatrix, x, α=1, β=0; kernel=:auto, threaded=true) -> y

Hermitian packed matrix–vector product (BLAS `hpmv`).
""" hpmv!

@doc raw"""
    spr!(P::PackedMatrix, x, α=1) -> P

Symmetric packed rank-1 update `P ← α·x·xᵀ + P` (BLAS `spr`).
""" spr!

@doc raw"""
    spr2!(P::PackedMatrix, x, y, α=1) -> P

Symmetric packed rank-2 update `P ← α·x·yᵀ + α·y·xᵀ + P` (BLAS `spr2`).
""" spr2!

@doc raw"""
    hpr!(P::PackedMatrix, x, α=1) -> P

Hermitian packed rank-1 update `P ← α·x·xᴴ + P`, `α` real (BLAS `hpr`).
""" hpr!

@doc raw"""
    hpr2!(P::PackedMatrix, x, y, α=1) -> P

Hermitian packed rank-2 update `P ← α·x·yᴴ + conj(α)·y·xᴴ + P` (BLAS `hpr2`).
""" hpr2!

@doc raw"""
    tpmv!(x, P::PackedMatrix; trans=:N, kernel=:auto) -> x

Triangular packed matrix–vector product `x ← op(P)·x` (BLAS `tpmv`); `P` must
have `kind=:triangular`.
""" tpmv!

@doc raw"""
    tpsv!(x, P::PackedMatrix; trans=:N, kernel=:auto) -> x

Triangular packed solve `op(P)·x = b` in place (BLAS `tpsv`).
""" tpsv!

@doc raw"""
    gbmv!(y, P::BandedMatrix, x, α=1, β=0; trans=:N, kernel=:auto, threaded=true) -> y

General banded matrix–vector product `y ← α·op(P)·x + β·y` (BLAS `gbmv`), only
the band is read.
""" gbmv!

@doc raw"""
    sbmv!(y, P::BandedMatrix, x, α=1, β=0; kernel=:auto, threaded=true) -> y

Symmetric banded matrix–vector product (BLAS `sbmv`).
""" sbmv!

@doc raw"""
    hbmv!(y, P::BandedMatrix, x, α=1, β=0; kernel=:auto, threaded=true) -> y

Hermitian banded matrix–vector product (BLAS `hbmv`).
""" hbmv!

@doc raw"""
    tbmv!(x, P::BandedMatrix; trans=:N, kernel=:auto) -> x

Triangular banded matrix–vector product `x ← op(P)·x` (BLAS `tbmv`).
""" tbmv!

@doc raw"""
    tbsv!(x, P::BandedMatrix; trans=:N, kernel=:auto) -> x

Triangular banded solve `op(P)·x = b` in place (BLAS `tbsv`).
""" tbsv!

@doc raw"""
    BandLUFactor{T}

Result of [`band_lu`](@ref): LU with partial pivoting in band storage (the
upper band grows by `kl` for the pivots). Solve with [`solve`](@ref).
""" BandLUFactor

@doc raw"""
    BandCholeskyFactor{T}

Result of [`band_cholesky`](@ref): Cholesky factor in band storage (`kd`
off-diagonals). Solve with [`solve`](@ref).
""" BandCholeskyFactor

@doc raw"""
    PackedCholeskyFactor{T}

Result of [`packed_cholesky`](@ref): the lower Cholesky factor in packed
storage (`n(n+1)/2` entries). Solve with [`solve`](@ref).
""" PackedCholeskyFactor

@doc raw"""
    AasenFactor{T}

Result of [`aasen_factor`](@ref): symmetric indefinite factorization
`A[p,p] = L·Tri·Lᴴ` with `Tri` tridiagonal (Aasen's method). Solve with
[`solve`](@ref).
""" AasenFactor

# ---- tensors -------------------------------------------------------------------------

@doc raw"""
    EinsumPlan

A contraction plan from [`plan_einsum`](@ref): the parsed specification, the
pairwise contraction order and its memory estimate. Reuse it with
[`einsum!`](@ref) to contract arrays of the same shapes without planning again.
""" EinsumPlan

@doc raw"""
    einsum!(C, plan, arrays...; α=1, β=0, workspace=nothing) -> C

In-place contraction `C ← α·einsum(arrays...) + β·C` with a precomputed
[`EinsumPlan`](@ref) (or a sliced plan from [`plan_einsum_sliced`](@ref)).
A reusable [`EinsumWorkspace`](@ref) avoids allocating the intermediates.

```julia
P = plan_einsum("abc,cd->abd", X, Y); ws = EinsumWorkspace(Float64)
einsum!(Z, P, X, Y; workspace=ws)
```
""" einsum!

@doc raw"""
    contract(A, B, axesA, axesB; conjugateA=false, conjugateB=false, output=:auto) -> Array

Tensor contraction over the axis pairs `(axesA[i], axesB[i])` (one GEMM after
a permutation). The result has the free axes of `A` then those of `B`. Sparse
operands (or `output=:sparse`) give a sparse result.

```julia
C = contract(A, B, (3,), (1,))     # Σ_k A[i,j,k]·B[k,l]
```
""" contract

@doc raw"""
    contract!(C, A, B, axesA, axesB; α=1, β=0) -> C

In-place [`contract`](@ref): `C ← α·contract(A, B, axesA, axesB) + β·C`.
""" contract!

@doc raw"""
    outer_product(A, B) -> Array

Tensor (outer) product: `C[i..., j...] = A[i...]·B[j...]`.
""" outer_product

@doc raw"""
    batched_matmul!(C, A, B, α=1, β=0; transA=:N, transB=:N) -> C

Batched matrix product over the trailing dimensions:
`C[:,:,k...] ← α·op(A[:,:,k...])·op(B[:,:,k...]) + β·C[:,:,k...]`, with
broadcasting of batch dimensions of size 1.
""" batched_matmul!

@doc raw"""
    TuckerFactor{T}

Result of [`hosvd`](@ref): `core` tensor and one factor matrix per mode;
[`reconstruct`](@ref) rebuilds the tensor.
""" TuckerFactor

@doc raw"""
    TensorTrainFactor{T}

Result of [`tensor_train`](@ref): the 3-D `cores` and the original `shape`;
[`reconstruct`](@ref) rebuilds the tensor.
""" TensorTrainFactor

@doc raw"""
    CPFactor{T,R}

Result of [`cp_als`](@ref): CP (CANDECOMP/PARAFAC) decomposition with
`weights`, one factor matrix per mode, `iterations`, `relative_error`,
`converged` and `reason`; [`reconstruct`](@ref) rebuilds the tensor.
""" CPFactor

@doc raw"""
    reconstruct(F::Union{TuckerFactor, TensorTrainFactor, CPFactor}) -> Array

Dense tensor represented by a Tucker, tensor-train or CP decomposition.
""" reconstruct

# ---- spectral ----------------------------------------------------------------------

@doc raw"""
    shift_invert(A; sigma=0, B=nothing, kwargs...) -> NamedTuple

Eigenvalues of `A` (or of the pencil `(A, B)`) nearest to `sigma`: implicitly
restarted Arnoldi ([`iram`](@ref)) on `(A - σB)⁻¹`, applied through one sparse
LU. Keywords go to `iram` (`nev`, `subspace`, `maxrestarts`…). Returns the
eigenvalues of `A`, the vectors and their residuals.
""" shift_invert

@doc raw"""
    polynomial_filter(A, low, high; degree=20, bounds=nothing, damping=:jackson) -> ChebyshevFilter

Chebyshev polynomial approximating the indicator of the interval `[low, high]`
of the spectrum of a Hermitian `A` (Jackson damping by default). Apply it with
[`apply_filter!`](@ref); used by [`eigenvalues_in_interval`](@ref).
""" polynomial_filter

@doc raw"""
    SparseSVDWorkspace(T, m, n, rank; subspace, blocksize, rowblock=64)
    SparseSVDWorkspace(A, rank; kwargs...)

Buffers of [`sparse_svd_lanczos`](@ref) for an `m×n` matrix and the given
`rank`: pass it as `workspace=` to repeated calls to avoid reallocating the
Krylov bases. Not shareable between concurrent calls.
""" SparseSVDWorkspace

@doc raw"""
    svd_snapshot(F::IncrementalSVDFactor) -> NamedTuple

Copy of the current state of an incremental SVD: `(; U, S, Vt, shape,
maxrank, updates, ...)`, independent of later updates.
""" svd_snapshot

@doc raw"""
    reorthogonalize_svd!(F::IncrementalSVDFactor; threaded=true) -> F

Restores the orthogonality of the singular vectors of an incremental SVD,
lost to rounding after many [`svd_add_rows!`](@ref)/[`svd_add_columns!`](@ref)
updates.
""" reorthogonalize_svd!

# ---- graphs ------------------------------------------------------------------------

@doc raw"""
    compare_orderings(A; algorithms=(:natural, :rcm, :minimum_degree, :multiple_minimum_degree, :multilevel_nested_dissection))

Computes each fill-reducing ordering and its metrics on the symmetric pattern
of `A`: one named tuple per algorithm with `permutation`, `factor_nnz`
(Cholesky fill), `fill`, `fill_ratio`, `bandwidth` and `profile`.
""" compare_orderings

@doc raw"""
    select_sparse_ordering(A; algorithms) -> (; best, comparisons)

The ordering of [`compare_orderings`](@ref) with the smallest Cholesky factor
(ties broken by bandwidth then profile), with all the comparisons.
""" select_sparse_ordering

@doc raw"""
    is_structurally_symmetric(A) -> Bool

Whether the pattern of the off-diagonal nonzero entries of `A` is symmetric
(values are not compared).
""" is_structurally_symmetric

# ---- utils -------------------------------------------------------------------------

@doc raw"""
    GemmTuning

Result of [`autotune_gemm`](@ref): the fastest validated `config`
([`GemmConfig`](@ref)), the `kernel`, the timings of all `candidates` and the
machine (`cpu`) they were measured on.
""" GemmTuning
