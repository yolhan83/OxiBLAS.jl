# SPDX-License-Identifier: Apache-2.0
# Docstrings of the sparse family: storage formats, factor types,
# preconditioners, utilities and Matrix Market I/O.

# ---- storage formats -----------------------------------------------------------

@doc raw"""
    CSRMatrix(m, n, rowptr, colind, values)
    CSRMatrix(A)

Compressed Sparse Row (CSR) matrix with 1-based indexing.

Row pointers `rowptr` have length `m + 1` with `rowptr[1] == 1` and monotonic
increasing values (`rowptr[end] == length(values) + 1`). Column indices `colind`
and coefficients `values` have length equal to the number of stored nonzeros.
Duplicate column entries within any row are rejected. Explicit stored zeros are
preserved.

Accepts real, complex and generic (e.g. `BigFloat`) scalars. Interoperates with
`SparseArrays.SparseMatrixCSC` and other OxiBLAS sparse formats.
""" CSRMatrix

@doc raw"""
    CSCMatrix(m, n, colptr, rowind, values)
    CSCMatrix(A)
    CSCMatrix(S::SparseMatrixCSC)

Compressed Sparse Column (CSC) matrix with validated 1-based indices.

Column pointers `colptr` have length `n + 1` with `colptr[1] == 1` and
`colptr[end] == length(values) + 1`. Row indices `rowind` are strictly increasing
and unique within each column. Explicit stored zeros are preserved.

Converting a `CSCMatrix` from or to a `SparseArrays.SparseMatrixCSC` shares storage
without copying when indices are valid and sorted.
""" CSCMatrix

@doc raw"""
    COOMatrix(m, n, rows, cols, values)
    COOMatrix(A)

Coordinate (COO) sparse matrix in canonical form.

Coordinates `rows` and `cols` use 1-based indexing. During canonical
construction, duplicate coordinates `(i, j)` are automatically summed, exact
zeros are removed, and entries are sorted in row-major order.
""" COOMatrix

@doc raw"""
    DIAMatrix(A::CSRMatrix)
    DIAMatrix(m, n, offsets, data)

Diagonal storage: `data[i,k]` is the entry `(i, i+offsets[k])`. Efficient for
matrices with a few full diagonals (stencils). Convertible from any OxiBLAS
sparse format or `SparseMatrixCSC`; usable with [`sparse_mul!`](@ref) and the
iterative solvers.
""" DIAMatrix

@doc raw"""
    ELLMatrix(A::CSRMatrix; width=max row length)
    ELLMatrix(m, n, indices, values)

ELLPACK storage: every row padded to `width` entries (`indices[i,k]`,
`values[i,k]`, padding with index 0). Regular access pattern for matrices with
rows of similar length.
""" ELLMatrix

@doc raw"""
    BSRMatrix(A::CSRMatrix; blocksize=(4,4))
    BSRMatrix(m, n, br, bc, rowptr, colind, blocks)

Block sparse row storage: dense `br×bc` blocks (`blocks[:,:,k]`) indexed like a
CSR matrix of blocks. Suited to matrices from vector-valued PDEs.
""" BSRMatrix

@doc raw"""
    BSCMatrix(A::CSRMatrix; blocksize=(4,4))

Block sparse column storage (the transpose of a [`BSRMatrix`](@ref) layout).
""" BSCMatrix

@doc raw"""
    HYBMatrix(A::CSRMatrix; width=average row length)

Hybrid storage: an [`ELLMatrix`](@ref) part of fixed `width` plus a
[`COOMatrix`](@ref) for the entries of the longer rows.
""" HYBMatrix

@doc raw"""
    SELLMatrix(A::CSRMatrix; sliceheight=8, sorting_window=1)

Sliced ELLPACK (SELL-C-σ): rows grouped in slices of `sliceheight`, each
padded to its own width, optionally sorted by length within windows of
`sorting_window` slices.
""" SELLMatrix

# ---- factor types ----------------------------------------------------------------

@doc raw"""
    SparseLUFactor{T,R}

Sparse LU `A[p,q] = L·U` (CSR factors), from [`sparse_lu`](@ref) or the
incomplete [`ilut`](@ref) (`drop_tolerance > 0`). Solve with
[`solve`](@ref); usable as a preconditioner.
""" SparseLUFactor

@doc raw"""
    SparseBlockedLUFactor{T,R}

Result of [`sparse_lu_blocked`](@ref): a [`SparseLUFactor`](@ref) computed by
dense panels. Solve with [`solve`](@ref).
""" SparseBlockedLUFactor

@doc raw"""
    SparseStaticLUFactor{T,R}

Result of [`sparse_lu_static`](@ref): LU with static pivoting, where tiny pivots
are replaced by `±epsilon` (`perturbations` records them) instead of being
exchanged. Pair it with iterative refinement; [`sparse_direct_diagnostics`](@ref)
reports whether it was perturbed.
""" SparseStaticLUFactor

@doc raw"""
    MultifrontalLUFactor{T,R}

Result of [`multifrontal_lu`](@ref): `A[p,q] = L·U` computed front by front
along the elimination tree, with threshold partial pivoting
(`pivot_threshold`) and the observed `growth_factor`. The engine behind
`OxiBLAS.lu` and `\` on sparse matrices.
""" MultifrontalLUFactor

@doc raw"""
    SparseCholeskyFactor{T,R}

Sparse Cholesky `A = L·Lᴴ` (CSC factor), from [`sparse_cholesky`](@ref) or the
incomplete [`ic0`](@ref)/[`ict`](@ref) (`incomplete = true`). Solve with
[`solve`](@ref); usable as a preconditioner for [`pcg`](@ref).
""" SparseCholeskyFactor

@doc raw"""
    MultifrontalCholeskyFactor{T}

Result of [`multifrontal_cholesky`](@ref) / [`supernodal_cholesky`](@ref):
`A[p,p] = L·Lᴴ` with the elimination tree (`parents`) and the supernodes
(`nodes`). The engine behind `OxiBLAS.cholesky` on sparse matrices.
""" MultifrontalCholeskyFactor

@doc raw"""
    SparseLDLFactor{T,R}

Result of [`sparse_ldlt`](@ref): sparse symmetric indefinite factorization
`A[p,p] = L·D·Lᴴ` with 1×1 and 2×2 pivots. Solve with [`solve`](@ref);
[`ldl_inertia`](@ref) counts the positive, negative and zero eigenvalues.
""" SparseLDLFactor

@doc raw"""
    SparseQRFactor{T,R}

Result of [`sparse_qr`](@ref): sparse Householder QR `A[:,p] = Q·R`, with `Q`
kept as sparse reflectors. [`solve`](@ref) / [`least_squares`](@ref) give the
least-squares solution; [`sparse_apply_q!`](@ref) applies `Q`.
""" SparseQRFactor

@doc raw"""
    OutOfCoreFactor{T}

Result of [`out_of_core_lu`](@ref) / [`out_of_core_cholesky`](@ref): a dense
factorization whose tiles live in a file, with at most `memory_budget` bytes in
memory. Solve with [`solve`](@ref); [`out_of_core_memory_usage`](@ref) reports
the memory used.
""" OutOfCoreFactor

@doc raw"""
    ILUFactor{T}

Result of [`ilu0`](@ref): incomplete LU with zero fill (the sparsity pattern of
`A`). A preconditioner for [`gmres`](@ref), [`bicgstab`](@ref)…
""" ILUFactor

# ---- preconditioners --------------------------------------------------------------

@doc raw"""
    ilut(A; droptol=1e-3, maxfill=20, kwargs...) -> SparseLUFactor

Incomplete LU with threshold dropping: entries below `droptol` (relative to
their row) are dropped and each row keeps at most `maxfill` extra entries.
Stronger than [`ilu0`](@ref) for nonsymmetric systems.

```julia
x = gmres(A, b; preconditioner=ilut(A)).x
```
""" ilut

@doc raw"""
    ic0(A; kwargs...) -> SparseCholeskyFactor

Incomplete Cholesky with zero fill, for Hermitian positive definite `A`: the
usual preconditioner of [`pcg`](@ref).

```julia
x = pcg(A, b; preconditioner=ic0(A)).x
```
""" ic0

@doc raw"""
    ict(A; droptol=1e-3, kwargs...) -> SparseCholeskyFactor

Incomplete Cholesky with threshold dropping (more fill than [`ic0`](@ref),
fewer iterations).
""" ict

@doc raw"""
    ainv(A; droptol=0.1, maxnnz=20) -> AINVPreconditioner

Approximate inverse preconditioner by biconjugation (AINV): sparse factors
`Z·D⁻¹·Wᴴ ≈ A⁻¹`, applied by sparse products only (no triangular solves, hence
parallel).
""" ainv

@doc raw"""
    ApproxInversePreconditioner{T}

Result of [`spai`](@ref): an explicit sparse approximate inverse `M ≈ A⁻¹`
(field `inverse`), applied as `z = M·r`.
""" ApproxInversePreconditioner

@doc raw"""
    relaxation_preconditioner(A; omega=1, sweeps=1, symmetric=false) -> RelaxationPreconditioner

Relaxation sweeps as a preconditioner: Gauss–Seidel (`omega = 1`), SOR
(`0 < omega < 2`), or the symmetric SSOR (`symmetric=true`, required by
[`pcg`](@ref) and [`minres`](@ref)). `sweeps` sweeps per application.
""" relaxation_preconditioner

@doc raw"""
    RelaxationPreconditioner{T,R}

Result of [`relaxation_preconditioner`](@ref).
""" RelaxationPreconditioner

@doc raw"""
    polynomial_preconditioner(A; degree=8, bounds=nothing, kind=:chebyshev) -> PolynomialPreconditioner

Polynomial preconditioner `p(A) ≈ A⁻¹` of the given `degree`: Chebyshev on the
spectral interval `bounds = (lo, hi)` (estimated by Gershgorin discs when
`nothing`) or a truncated Neumann series (`kind=:neumann`). Applied by sparse
products only.
""" polynomial_preconditioner

@doc raw"""
    PolynomialPreconditioner{T,R}

Result of [`polynomial_preconditioner`](@ref).
""" PolynomialPreconditioner

@doc raw"""
    additive_schwarz(A; subdomains=nthreads(), overlap=1, solver=:cholesky, threaded=true) -> AdditiveSchwarzPreconditioner

Overlapping additive Schwarz (domain decomposition): `A` is split into
`subdomains` contiguous blocks extended by `overlap` rows; each local block is
factorized (`solver` = `:cholesky`, `:lu` or `:ilu0`) and the local solves are
added, in parallel.
""" additive_schwarz

@doc raw"""
    AdditiveSchwarzPreconditioner{T,R,F}

Result of [`additive_schwarz`](@ref).
""" AdditiveSchwarzPreconditioner

@doc raw"""
    AMGPreconditioner{T,R}

Result of [`amg`](@ref): an algebraic multigrid hierarchy (aggregation,
optionally smoothed) applied as one V- or W-cycle per preconditioner call.
""" AMGPreconditioner

# ---- operations and utilities -------------------------------------------------------

@doc raw"""
    sparse_mul!(y::AbstractVector, A, x::AbstractVector, α=1, β=0; trans=:N, threaded=true, parallel_threshold=262144) -> y
    sparse_mul!(Y::AbstractMatrix, A, X::AbstractMatrix, α=1, β=0; trans=:N, threaded=true, parallel_threshold=262144) -> Y

In-place sparse matrix-vector product `y ← α·op(A)·x + β·y` (SpMV) or sparse matrix-matrix
product `Y ← α·op(A)·X + β·Y` (SpMM).

# Arguments
- `y` / `Y`: Destination dense vector or matrix. Must not alias sparse storage of `A`.
- `A`: Sparse matrix ([`CSRMatrix`](@ref), [`CSCMatrix`](@ref), [`COOMatrix`](@ref), extended formats, or `SparseMatrixCSC`).
- `x` / `X`: Input dense vector or matrix.
- `α`, `β`: Scalar coefficients (default: `α = 1`, `β = 0`).
- `trans`: Operation on `A`: `:N` for $A$, `:T` for $A^{\top}$, `:C` for $A^{\mathrm{H}}$.
- `threaded`: Enable multi-threading (default: `true`).
- `parallel_threshold`: Workload threshold before spawning tasks across threads (default: `262144`).

Multi-threading partitions rows (for CSR SpMV or transposed CSC SpMV) or columns (for SpMM)
across worker tasks.
""" sparse_mul!

@doc raw"""
    sparse_matmul(A::CSRMatrix, B::CSRMatrix; droptol=0, threaded=true, parallel_threshold=30_000) -> CSRMatrix
    sparse_matmul(A::CSCMatrix, B::CSCMatrix; droptol=0, threaded=true, parallel_threshold=30_000) -> CSCMatrix
    sparse_matmul(A, B; kwargs...)

Native Gustavson sparse matrix-matrix multiplication (`C = A * B`).

Computes the sparse product without converting to dense arrays or delegating to
external libraries. CSR×CSR uses row-wise accumulation with an $O(n)$ sparse
accumulator per thread; CSC×CSC uses column-wise accumulation with an $O(m)$
sparse accumulator per thread. Output row/column indices are strictly sorted.

# Arguments
- `A`, `B`: Sparse matrices ([`CSRMatrix`](@ref), [`CSCMatrix`](@ref), [`COOMatrix`](@ref), or `SparseArrays.SparseMatrixCSC`).
- `droptol::Real`: Drop tolerance (default: `0`). Elements with `abs(val) <= droptol` are omitted.
- `threaded::Bool`: Enable multi-threading (default: `true`).
- `parallel_threshold::Integer`: Heuristic workload threshold (`nnz(A) * nnz(B) / dim`) for thread spawning (default: `30_000`).

When multi-threading is active, work is adaptively partitioned into chunked tasks
with thread-private accumulators and assembled into the final compressed structure with
zero allocation overhead.
""" sparse_matmul

@doc raw"""
    sparse_add(A, B, α=1, β=1) -> CSRMatrix

Sparse linear combination `α·A + β·B` (same size and element type), with
duplicate entries merged.
""" sparse_add

@doc raw"""
    sparse_transpose(A; conjugate=false) -> CSRMatrix

Explicit transpose (or adjoint with `conjugate=true`) as a new CSR matrix.
""" sparse_transpose

@doc raw"""
    sparse_permute(A, p, q=p) -> CSRMatrix

Permuted copy `A[p, q]` (symmetric permutation `A[p, p]` by default), for
applying an ordering such as [`reverse_cuthill_mckee`](@ref).
""" sparse_permute

@doc raw"""
    sparse_diagonal(values; offset=0, m, n) -> CSRMatrix

Sparse matrix with `values` on the diagonal `offset` (0: main diagonal,
positive: above). The size defaults to the smallest one that holds the
diagonal.
""" sparse_diagonal

@doc raw"""
    sparse_norm(A, kind=:fro)

Norm of a sparse matrix from its stored entries: `:fro`, `:max`, `:one`
(column sums) or `:inf` (row sums).
""" sparse_norm

@doc raw"""
    sparse_apply_q!(B, F::SparseQRFactor; trans=:N) -> B

Applies the orthogonal factor of a sparse QR to the columns of `B` in place:
`Q·B` (`trans=:N`) or `Qᴴ·B` (`trans=:C`).
""" sparse_apply_q!

@doc raw"""
    sparse_svd(A, rank; oversampling=8, power_iterations=2, seed, threaded=true) -> SVDFactor

Randomized partial SVD of a sparse matrix: the leading `rank` singular
triplets from a random range finder with `power_iterations` refinement steps.
Fast for low ranks; for an accurate partial SVD with convergence control use
[`sparse_svd_lanczos`](@ref).
""" sparse_svd

@doc raw"""
    supernodal_cholesky(A; kwargs...) -> MultifrontalCholeskyFactor

[`multifrontal_cholesky`](@ref) with supernodes forced on (dense blocks for
columns with identical structure).
""" supernodal_cholesky

@doc raw"""
    sparse_direct_diagnostics(A, F, b, x=solve(F, b)) -> NamedTuple

Quality check of a sparse direct solve: `(; residual_norms,
componentwise_backward_error, finite, perturbed)`, one entry per right-hand
side; `perturbed` is true when `F` is a [`SparseStaticLUFactor`](@ref) that
replaced pivots.
""" sparse_direct_diagnostics

@doc raw"""
    out_of_core_memory_usage(F::OutOfCoreFactor) -> NamedTuple

Memory accounting of an out-of-core factorization: `(; payload_bytes,
budget_bytes, metadata_bytes, includes_input_output, includes_runtime_overhead)`.
""" out_of_core_memory_usage

# ---- graphs and orderings -----------------------------------------------------------

@doc raw"""
    reverse_cuthill_mckee(A) -> Vector{Int}

Reverse Cuthill–McKee ordering of the (symmetrized) graph of `A`: a
permutation that reduces the bandwidth and the profile. Apply it with
[`sparse_permute`](@ref).
""" reverse_cuthill_mckee

@doc raw"""
    minimum_degree_ordering(A) -> Vector{Int}

Exact minimum degree ordering on the quotient graph (supervariables, mass
elimination): a fill-reducing permutation for sparse Cholesky and LU. See also
[`amd_ordering`](@ref) (approximate, faster) and
[`multiple_minimum_degree`](@ref).
""" minimum_degree_ordering

@doc raw"""
    connected_components(A) -> (; labels, components, count)

Connected components of the (symmetrized) graph of `A`: `labels[i]` is the
component of vertex `i`, `components` lists the vertices of each one.
""" connected_components

@doc raw"""
    level_sets(A, start) -> Vector{Vector{Int}}

Breadth-first level structure of the graph of `A` from vertex `start`
(level 1 is `[start]`).
""" level_sets

@doc raw"""
    bandwidth_profile(A) -> (; lower, upper, bandwidth, profile)

Lower and upper bandwidths, total bandwidth `lower + upper + 1` and profile
(envelope size) of `A`, counting the stored nonzero entries.
""" bandwidth_profile

# ---- Matrix Market -------------------------------------------------------------------

@doc raw"""
    read_matrix_market(path_or_io; T=nothing, format=:csr)

Reads a Matrix Market file (`coordinate` or `array`; `real`, `integer`,
`complex` or `pattern`; `general`, `symmetric`, `hermitian` or
`skew-symmetric`). Symmetric storage is expanded. `T` overrides the element
type (default `Float64`, or `ComplexF64` for complex files); `format` selects
the result: `:csr`, `:csc`, `:coo` (OxiBLAS formats) or `:dense` (`Matrix`).
""" read_matrix_market

@doc raw"""
    write_matrix_market(path_or_io, A; symmetry=:general)

Writes a sparse matrix in Matrix Market coordinate format. With
`symmetry=:symmetric`, `:hermitian` or `:skew` (square matrices) only the lower
triangle is written (strictly lower for `:skew`).
""" write_matrix_market
