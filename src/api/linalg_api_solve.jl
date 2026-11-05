# SPDX-License-Identifier: Apache-2.0
# Linear solves: OxiBLAS.ldiv / OxiBLAS.ldiv!, and `\` / LinearAlgebra.ldiv!
# on the OxiBLAS sparse formats.
#
# - OxiBLAS.ldiv and OxiBLAS.ldiv! are OxiBLAS's own functions, distinct from
#   LinearAlgebra.ldiv! (the module never does `using LinearAlgebra`). They
#   take any matrix that OxiBLAS.factorize accepts, or any Factorization.
# - The LinearAlgebra extensions only cover signatures that contain an OxiBLAS
#   sparse format, as in linalg_api_sparse.jl.
# - Every solve ends in the existing primitive LinearAlgebra.ldiv!(F, X) of a
#   factorization; nothing here solves anything itself.

# ---- O(nnz) structure tests on the OxiBLAS formats -----------------------------
# Without them, LinearAlgebra's generic istril/istriu (and so its generic `\`)
# read all n² positions through getindex. As in SparseArrays, a stored zero
# does not count.
function _any_stored(pred::F,A::CSCMatrix) where F
    @inbounds for j in 1:A.n, q in A.colptr[j]:A.colptr[j+1]-1
        pred(Int(A.rowind[q]),j,A.values[q]) && return true
    end
    false
end
function _any_stored(pred::F,A::CSRMatrix) where F
    @inbounds for i in 1:A.m, q in A.rowptr[i]:A.rowptr[i+1]-1
        pred(i,Int(A.colind[q]),A.values[q]) && return true
    end
    false
end
function _any_stored(pred::F,A::COOMatrix) where F
    @inbounds for q in eachindex(A.values)
        pred(Int(A.rows[q]),Int(A.cols[q]),A.values[q]) && return true
    end
    false
end
_any_stored(pred::F,A::ExtendedSparse) where F=_any_stored(pred,CSCMatrix(A))
_LA.istril(A::_OxiSparse,k::Integer=0)=!_any_stored((i,j,v)->j-i>k && !iszero(v),A)
_LA.istriu(A::_OxiSparse,k::Integer=0)=!_any_stored((i,j,v)->j-i<k && !iszero(v),A)

# ---- sparse triangular matrices: TriangularEngine over the native kernel --------
# The engine keeps its own copy in CSR or CSC (the two storages of
# sparse_triangular_solve, which reads only the triangle it is asked for);
# `src` is the matching LinearAlgebra triangular wrapper, as for dense input.
_sp_tri_storage(A::Union{CSRMatrix,CSCMatrix},::Type{T}) where T=eltype(A)===T ? copy(A) : _mul_convert(T,A)
_sp_tri_storage(A::SparseMatrixCSC,::Type{T}) where T=CSCMatrix(SparseMatrixCSC{T}(copy(A)))
_sp_tri_storage(A::Union{COOMatrix,ExtendedSparse},::Type{T}) where T=CSCMatrix(_mul_convert(T,A))
function _sp_tri_engine(W::Type,A,upper::Bool,unit::Bool)
    S=_sp_tri_storage(A,_la_eltype(eltype(A)))
    TriangularEngine(W(S),S,upper,unit)
end
_tri_kernel!(K::Union{CSRMatrix,CSCMatrix},X::AbstractVecOrMat,E,tr)=
    _sparse_triangular_solve!(X,K;uplo=E.upper ? :U : :L,trans=tr,unit=E.unit)
# Unit lower factor A·D⁻¹ (F.L of the lower-triangular engine), on the
# stored entries.
function _unit_lower(A::_LA.LowerTriangular{<:Any,<:CSCMatrix},d)
    M=copy(parent(A))
    @inbounds for j in 1:M.n, q in M.colptr[j]:M.colptr[j+1]-1; M.values[q]/=d[j]; end
    _LA.UnitLowerTriangular(M)
end
function _unit_lower(A::_LA.LowerTriangular{<:Any,<:CSRMatrix},d)
    M=copy(parent(A))
    @inbounds for q in eachindex(M.values); M.values[q]/=d[M.colind[q]]; end
    _LA.UnitLowerTriangular(M)
end

const _SpTriWrap=Union{_LA.UpperTriangular{<:Any,<:_SpIn},_LA.UnitUpperTriangular{<:Any,<:_SpIn},
                       _LA.LowerTriangular{<:Any,<:_SpIn},_LA.UnitLowerTriangular{<:Any,<:_SpIn}}
_sp_tri_upper(A)=A isa Union{_LA.UpperTriangular,_LA.UnitUpperTriangular}
_sp_tri_unit(A)=A isa Union{_LA.UnitUpperTriangular,_LA.UnitLowerTriangular}
_sp_tri_wrapper(A)=A isa _LA.UpperTriangular ? _LA.UpperTriangular : A isa _LA.UnitUpperTriangular ? _LA.UnitUpperTriangular :
                   A isa _LA.LowerTriangular ? _LA.LowerTriangular : _LA.UnitLowerTriangular
"""
    OxiBLAS.factorize(UpperTriangular(A::sparse)) (and the lower/unit wrappers)

`LU` with a `TriangularEngine` over the native sparse substitution
(`sparse_triangular_solve`): no elimination, and only the wrapped triangle
is read. The engine keeps its own copy of `A`.
"""
factorize(A::_SpTriWrap)=(_LA.checksquare(A); LU(_sp_tri_engine(_sp_tri_wrapper(A),parent(A),_sp_tri_upper(A),_sp_tri_unit(A))))

# ---- OxiBLAS.ldiv / OxiBLAS.ldiv! ----------------------------------------------
function ldiv end
function ldiv! end

# Sparse right-hand sides are densified once (the solution is dense); other
# arrays are read as they are.
_dense_rhs(B::Union{SparseArrays.AbstractSparseArray,SparseTensor})=Array(B)
_dense_rhs(B::AbstractVecOrMat)=B
_solve_eltype(F,B)=(T=Base.promote_op(\,eltype(F),eltype(B)); isconcretetype(T) ? T : promote_type(eltype(F),eltype(B)))

"""
    OxiBLAS.ldiv!(Y, F::Factorization, B; workspace=nothing) -> Y
    OxiBLAS.ldiv!(Y, A::AbstractMatrix, B; workspace=nothing) -> Y

Solves `A*X = B` and writes `X` into `Y`, which has `size(A, 2)` rows and as
many columns as `B`.

- `F` is any factorization with a `LinearAlgebra.ldiv!(F, X)` method: the
  OxiBLAS ones, and their `transpose`/`adjoint`. It is reused as it is.
- `A` is factorized first with `OxiBLAS.factorize` (dense, structured or
  sparse input). An input that it does not accept gets a MethodError.
- `B` is never modified, except with `Y === B` on a square system, which
  solves in place. A `Y` that only partially overlaps `B` is rejected.
- A tall `F` (`m > n`) gives the least-squares solution, a wide one (`n > m`)
  the solution of that factorization's own `ldiv!` (the minimum-norm one for
  dense `QR`/`LQ`); the sparse QR has no wide solve.
- **`workspace` (OxiBLAS extension beyond `LinearAlgebra.ldiv!`).**
  - **Why:** For a tall orthogonal factorization (`QR`, `QRPivoted`,
    `QRSparse`, or `LQ'` with `m > n`), the in-place primitive
    `LinearAlgebra.ldiv!(F, X)` applies `Qᴴ` to an `m`-row buffer before
    extracting the `n` solution rows. Because `Y` only has `n < m` rows and `B`
    must not be overwritten, an `m`-row working buffer `X` is required.
  - **When you do NOT need it:** On square systems (`m == n`), wide systems
    (`n > m`, where `Y` itself has `n > m` rows and serves as the working
    buffer), `SVD` factorizations (which project into rank-sized scratch and
    write directly into `Y`), or when `B` is sparse (whose dense copy is
    reused in place), `OxiBLAS.ldiv!` already avoids allocating `X` with
    `workspace=nothing`.
  - **When and how to use it:** When solving repeated tall least-squares
    systems (`m > n`) in a hot loop with a dense `B` and `QR` / `QRPivoted` /
    `QRSparse` / `LQ'`, preallocate `ws = Vector{T}(undef, m)` (or
    `Matrix{T}(undef, m, nrhs)`) once per task and pass `workspace=ws` to
    eliminate the `m`-row buffer allocation. `workspace` must have at least
    `(max(m, n), size(B, 2))` elements, 1-based indexing, must not alias `Y`
    or `B`, and must be exclusive to the calling task during the call.
- `Y` must be a dense array: a solution is dense. A sparse `B` is read
  through a dense copy.
- The solve runs in the factor's element type. A real factor and a complex
  `B` of the same precision are solved as two real systems.
"""
function ldiv!(Y::AbstractVecOrMat,F::_LA.Factorization,B::AbstractVecOrMat;workspace=nothing)
    Base.require_one_based_indexing(Y,B)
    m,n=size(F)
    size(B,1)==m || throw(DimensionMismatch("factorization has $m rows, right-hand side has $(size(B,1))"))
    size(Y,1)==n && size(Y,2)==size(B,2) ||
        throw(DimensionMismatch("destination has size $(size(Y)), the solution has size ($n, $(size(B,2)))"))
    Y isa Union{SparseArrays.AbstractSparseArray,SparseTensor} &&
        throw(ArgumentError("the destination of a solve must be a dense array"))
    if Y===B
        m==n || throw(ArgumentError("an in-place solve (Y === B) requires a square system"))
        return (_LA.ldiv!(F,Y); Y)
    end
    Base.mightalias(Y,B) && throw(ArgumentError("Y and B overlap; pass the same array twice to solve in place"))
    Bd=_dense_rhs(B)
    if m==n
        copyto!(Y,Bd); _LA.ldiv!(F,Y)
    elseif F isa SVD && eltype(Y)===eltype(F) && eltype(Bd)===eltype(F) && Y isa StridedVecOrMat && Bd isa StridedVecOrMat
        _svd_solve!(Y,_eng(F),Bd)
    elseif n>m && workspace===nothing && eltype(Y)===_solve_eltype(F,B) && Y isa StridedVecOrMat
        fill!(Y,zero(eltype(Y)))
        copyto!(view(Y,1:m,:),Bd); _LA.ldiv!(F,Y)
    else
        # The primitive ldiv!(F, X) reads and writes max(m,n) rows of X.
        rows=max(m,n); T=_solve_eltype(F,B)
        X=if workspace!==nothing
            Base.require_one_based_indexing(workspace)
            (size(workspace,1)>=rows && size(workspace,2)>=size(B,2)) ||
                throw(DimensionMismatch("workspace has size $(size(workspace)), need at least ($rows, $(size(B,2)))"))
            (Base.mightalias(workspace,Y) || Base.mightalias(workspace,B)) &&
                throw(ArgumentError("workspace must not alias Y or B"))
            W=B isa AbstractVector ? view(workspace,1:rows) : view(workspace,1:rows,1:size(B,2))
            rows>m && fill!(view(W,m+1:rows,:),zero(eltype(W)))
            W
        elseif B!==Bd && m>=n && eltype(Bd)===T
            Bd
        elseif m>=n
            B isa AbstractVector ? Vector{T}(undef,rows) : Matrix{T}(undef,rows,size(B,2))
        else
            B isa AbstractVector ? zeros(T,rows) : zeros(T,rows,size(B,2))
        end
        X===Bd || copyto!(view(X,1:m,:),Bd)
        _LA.ldiv!(F,X)
        copyto!(Y,view(X,1:n,:))
    end
    Y
end
ldiv!(Y::AbstractVecOrMat,A::AbstractMatrix,B::AbstractVecOrMat;workspace=nothing)=ldiv!(Y,factorize(A),B;workspace)

"""
    OxiBLAS.ldiv(F::Factorization, B) -> X
    OxiBLAS.ldiv(A::AbstractMatrix, B) -> X

Allocating form of `OxiBLAS.ldiv!`. `X` is a `Vector` or a `Matrix`, like
`B`, with element type `promote_op(\\, eltype(F), eltype(B))`.
"""
function ldiv(F::_LA.Factorization,B::AbstractVecOrMat)
    m,n=size(F); T=_solve_eltype(F,B)
    if m>n && !(F isa SVD) && B isa AbstractVector
        Base.require_one_based_indexing(B)
        length(B)==m || throw(DimensionMismatch("factorization has $m rows, right-hand side has $(length(B))"))
        Bd=_dense_rhs(B)
        X=B!==Bd && eltype(Bd)===T && Bd isa Vector{T} ? Bd : copyto!(Vector{T}(undef,m),Bd)
        _LA.ldiv!(F,X)
        return resize!(X,n)
    end
    ldiv!(B isa AbstractVector ? Vector{T}(undef,n) : Matrix{T}(undef,n,size(B,2)),F,B)
end
ldiv(A::AbstractMatrix,B::AbstractVecOrMat)=ldiv(factorize(A),B)

# ---- `\` and LinearAlgebra.ldiv! on the OxiBLAS sparse formats -----------------
# The factorization is chosen by OxiBLAS.factorize, as SparseArrays does for
# SparseMatrixCSC: triangular -> substitution, Hermitian -> Cholesky or
# Bunch–Kaufman, square -> LU, tall -> QR. These methods replace the generic
# AbstractMatrix ones, whose istril/istriu and triangular solves read A
# through getindex.
const _OxiSolveIn=Union{_OxiSparse,Transpose{<:Any,<:_OxiSparse},Adjoint{<:Any,<:_OxiSparse},
                        _LA.Hermitian{<:Any,<:_OxiSparse},_LA.Symmetric{<:Any,<:_OxiSparse},
                        _LA.UpperTriangular{<:Any,<:_OxiSparse},_LA.UnitUpperTriangular{<:Any,<:_OxiSparse},
                        _LA.LowerTriangular{<:Any,<:_OxiSparse},_LA.UnitLowerTriangular{<:Any,<:_OxiSparse}}
# Separate vector and matrix methods: LinearAlgebra's triangular and
# Hermitian `\` are split the same way, so a single AbstractVecOrMat method
# would be ambiguous with them.
Base.:\(A::_OxiSolveIn,B::AbstractVector)=ldiv(A,B)
Base.:\(A::_OxiSolveIn,B::AbstractMatrix)=ldiv(A,B)
_LA.ldiv!(Y::AbstractVecOrMat,A::_OxiSolveIn,B::AbstractVecOrMat)=ldiv!(Y,A,B)
# SparseArrays 1.10 defines `\` for each triangular wrapper and a sparse
# vector (AbstractCompressedVector, an internal type that later versions may
# not have): one method per wrapper, with the element types it requires, is
# more specific than both. Resolved when the package is compiled: no cost.
@static if isdefined(SparseArrays,:AbstractCompressedVector)
    for W in (:UpperTriangular,:UnitUpperTriangular,:LowerTriangular,:UnitLowerTriangular)
        @eval Base.:\(A::_LA.$W{<:Number,<:_OxiSparse},B::SparseArrays.AbstractCompressedVector{<:Number})=ldiv(A,B)
    end
end
"""
    LinearAlgebra.ldiv!(A::OxiBLAS-sparse, B) -> B

Overwrites `B` with the solution of the square system `A*X = B`; `A` is
factorized (`OxiBLAS.factorize`) and not modified.
"""
_LA.ldiv!(A::_OxiSolveIn,B::AbstractVecOrMat)=(_LA.checksquare(A); ldiv!(B,A,B))
