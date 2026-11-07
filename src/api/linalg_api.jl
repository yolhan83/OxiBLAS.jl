# SPDX-License-Identifier: Apache-2.0
# High-level API in the style of LinearAlgebra, built over the native engines.
#
# - Factorization types are subtypes of LinearAlgebra.Factorization{T} and wrap
#   the engine objects without copying them.
# - OxiBLAS.lu, OxiBLAS.qr, ... are separate functions from LinearAlgebra's.
#   None of them is exported.
# - Extensions of LinearAlgebra and Base functions only cover signatures that
#   contain an OxiBLAS type.
# - Each type provides only the primitives listed below. LinearAlgebra's
#   generic code builds the rest: `\`, ldiv!(Y,F,B), inv, det, logdet, the
#   adjoint/transpose wrappers, `/`, and AbstractQ products and Matrix(Q).
#
# The primitives:
#   - size;
#   - ldiv!(F,B) and ldiv! on AdjointFactorization/TransposeFactorization,
#     only for factorizations that solve systems;
#   - rdiv!(B,F), for which there is no generic method;
#   - logabsdet, issuccess;
#   - getproperty/propertynames;
#   - for an implicit Q: lmul!/rmul! (plain and adjoint) and det.
#
# Each specialization beyond these primitives says why it exists. The design
# and coverage tables are in docs/src/guide/highlevel.md.

const _Fact=_LA.Factorization
const _AdjF=_LA.AdjointFactorization
const _TrF=_LA.TransposeFactorization
const _AdjQ=_LA.AdjointQ
const _BlasReal=Union{Float32,Float64}   # LinearAlgebra.BlasReal, for disambiguation

# ---- accepted inputs -----------------------------------------------------------
# Standard dense matrices and the LinearAlgebra wrappers around them. Any
# other matrix type gets a MethodError, never a numeric fallback. Sparse
# inputs have their own methods in linalg_api_sparse.jl, so they are never
# densified.
const _DenseHS=Union{_LA.Hermitian{<:Any,<:StridedMatrix},_LA.Symmetric{<:Any,<:StridedMatrix}}
const _DenseTri=Union{_LA.UpperTriangular{<:Any,<:StridedMatrix},_LA.LowerTriangular{<:Any,<:StridedMatrix},
                      _LA.UnitUpperTriangular{<:Any,<:StridedMatrix},_LA.UnitLowerTriangular{<:Any,<:StridedMatrix}}
const _DenseAT=Union{Transpose{<:Any,<:StridedMatrix},Adjoint{<:Any,<:StridedMatrix}}
const _DenseIn=Union{StridedMatrix,_DenseHS,_DenseTri,_DenseAT}

# Element type of the factors, following LinearAlgebra's convention
# (Int -> Float64).
_la_eltype(::Type{T}) where T=typeof(zero(T)/oneunit(T))
# Copy into a plain matrix of a floating type. The engines copy their input,
# so a StridedMatrix that is already floating passes through unchanged.
_dense_input(A::StridedMatrix{<:GenericFloat})=A
_dense_input(A::AbstractMatrix)=Matrix{_la_eltype(eltype(A))}(A)

# ---- shared helpers for the primitives -----------------------------------------
_eng(F)=getfield(F,:engine)

function _check_rows(F,B,rows=size(F,1))
    Base.require_one_based_indexing(B)
    size(B,1)==rows || throw(DimensionMismatch("factorization has $rows rows, right-hand side has $(size(B,1))"))
    nothing
end

# Runs the in-place solve f!(X) on an X with the factor element type T, which
# the kernels require:
#   - B itself when its element type is T and it is strided;
#   - for a real factor and a complex B of the same precision, the real and
#     imaginary parts, solved separately (no complex copy of the factors);
#   - otherwise a converted copy, written back (an InexactError if the result
#     does not fit in B, as for in-place LinearAlgebra solves).
function _apply_split!(f!::F,::Type{T},B::AbstractVecOrMat) where {F,T}
    S=eltype(B)
    if S===T && B isa StridedVecOrMat
        f!(B)
    elseif T<:Real && S===Complex{T}
        Re=real.(B); Im=imag.(B); f!(Re); f!(Im)
        B .= complex.(Re,Im)
    else
        X=Array{T}(B); f!(X); copyto!(B,X)
    end
    B
end
_asmatrix(B::AbstractVector)=Base.ReshapedArray(B,(length(B),1),())
_asmatrix(B::AbstractMatrix)=B

# Row permutation of every column: X <- X[p,:], or X[p,:] <- X when inverse.
function _rowperm!(X::AbstractVecOrMat,p::AbstractVector{Int},inverse::Bool)
    t=similar(X,size(X,1))
    @inbounds for j in axes(X,2)
        c=view(X,:,j)
        if inverse
            for i in eachindex(p); t[p[i]]=c[i]; end
        else
            for i in eachindex(p); t[i]=c[p[i]]; end
        end
        copyto!(c,t)
    end
    X
end

# The same row permutation from the LAPACK interchange sequence (laswp):
# applying pivots[j] in order gives X[p,:], in reverse order its inverse.
# In place, no buffer.
function _rowswap!(X::AbstractVecOrMat,pivots::AbstractVector{Int},inverse::Bool)
    @inbounds for col in axes(X,2)
        c=view(X,:,col)
        for k in eachindex(pivots)
            j=inverse ? length(pivots)+1-k : k
            q=pivots[j]
            q!=j && ((c[j],c[q])=(c[q],c[j]))
        end
    end
    X
end

_trisolve!(X::AbstractVector,A,uplo,trans,unit)=trsv!(X,A;uplo,trans,unit)
_trisolve!(X::AbstractMatrix,A,uplo,trans,unit)=trsm!(X,A;uplo,trans,unit)

function _isoddperm(p::AbstractVector{Int})
    seen=falses(length(p)); cycles=0
    for i in eachindex(p)
        seen[i] && continue
        cycles+=1; j=i
        while !seen[j]; seen[j]=true; j=p[j]; end
    end
    isodd(length(p)-cycles)
end
# P with P*A == A[p,:].
function _permmatrix(::Type{T},p::AbstractVector{Int}) where T
    n=length(p); P=zeros(T,n,n)
    for i in 1:n; P[i,p[i]]=one(T); end
    P
end
function _triu_rows(a::AbstractMatrix,r::Int)
    R=zeros(eltype(a),r,size(a,2))
    @inbounds for j in axes(a,2), i in 1:min(r,j); R[i,j]=a[i,j]; end
    R
end

# ==== function declarations (OxiBLAS's own, distinct from LinearAlgebra's) ======
for f in (:lu,:lu!,:cholesky,:cholesky!,:ldlt,:ldlt!,:bunchkaufman,:bunchkaufman!,:qr,:qr!,
          :lq,:lq!,:svd,:svd!,:svdvals,:svdvals!,:eigen,:eigen!,:eigvals,:eigvals!,:eigvecs,
          :schur,:schur!,:hessenberg,:hessenberg!,:factorize,:det,:logdet,:logabsdet,
          :rank,:cond,:pinv,:nullspace,:isposdef,:issuccess,:sylvester,:lyap,
          :norm,:opnorm,:kron,:eigmax,:eigmin,:ordschur,:ordschur!,:spdiagm,:sparsevec)
    @eval function $f end
end

# ==== LU ========================================================================
"""
    OxiBLAS.LU{T,E} <: LinearAlgebra.Factorization{T}

LU factorization with row pivoting, which wraps an engine of type `E`:

- `LUFactor` (dense): `F.L*F.U == A[F.p,:]`;
- `MultifrontalLUFactor` (sparse): `F.L*F.U == A[F.p,F.q]`.

The type is mutable so that `lu!(F, A)` can refactorize in place.
"""
mutable struct LU{T,E} <: _Fact{T}
    engine::E
end
LU(E::LUFactor{T}) where T=LU{T,typeof(E)}(E)

Base.size(F::LU{<:Any,<:LUFactor})=size(_eng(F).factors)
_LA.issuccess(F::LU{<:Any,<:LUFactor})=_eng(F).info==0
function _lu_dense_ready(F)
    n=_LA.checksquare(F); E=_eng(F)
    E.info==0 || throw(SingularException(E.info))
    n,E
end
function _LA.ldiv!(F::LU{T,<:LUFactor},B::AbstractVecOrMat) where T
    n,E=_lu_dense_ready(F); _check_rows(F,B,n)
    _apply_split!(T,B) do X
        _rowswap!(X,E.pivots,false)
        _trisolve!(X,E.factors,:L,:N,true); _trisolve!(X,E.factors,:U,:N,false)
    end
end
# Aᴴ = Uᴴ·Lᴴ·P and Aᵀ = Uᵀ·Lᵀ·P, where P*A = A[p,:].
for (W,tr) in ((:_AdjF,:(:C)),(:_TrF,:(:T)))
    @eval function _LA.ldiv!(Fw::$W{<:Any,<:LU{T,<:LUFactor}},B::AbstractVecOrMat) where T
        n,E=_lu_dense_ready(parent(Fw)); _check_rows(Fw,B,n)
        _apply_split!(T,B) do X
            _trisolve!(X,E.factors,:U,$tr,false); _trisolve!(X,E.factors,:L,$tr,true)
            _rowswap!(X,E.pivots,true)
        end
    end
end
function _LA.logabsdet(F::LU{T,<:LUFactor}) where T
    n=_LA.checksquare(F); a=_eng(F).factors; d=zero(real(T)); s=one(T)
    for i in 1:n
        u=a[i,i]; d+=log(abs(u)); s*=sign(u)
    end
    (d,_isoddperm(_eng(F).p) ? -s : s)
end
function Base.getproperty(F::LU{T,<:LUFactor},s::Symbol) where T
    E=_eng(F)
    s===:L && return E.L
    s===:U && return E.U
    s===:p && return E.p
    s===:P && return _permmatrix(T,E.p)
    s===:factors && return E.factors
    s===:ipiv && return E.pivots
    s===:info && return E.info
    getfield(F,s)
end
Base.propertynames(F::LU{<:Any,<:LUFactor},private::Bool=false)=
    (:L,:U,:p,:P,(private ? (:factors,:ipiv,:info,:engine) : ())...)

"""
    OxiBLAS.lu(A, [RowMaximum()]; check=true) -> OxiBLAS.LU

LU factorization with partial pivoting (`lu_factor`). This function is not
`LinearAlgebra.lu`.

- `check=false` returns a factorization whose `issuccess` is `false` when A
  is singular.
- `lu!(A)` factors in place: `A` (a `Matrix` or a strided view of floating
  type) is overwritten by the factors and becomes `F.factors`, with no copy.
"""
lu(A::_DenseIn,::_LA.RowMaximum=_LA.RowMaximum();check::Bool=true)=LU(lu_factor(_dense_input(A);check))
lu!(A::StridedMatrix{<:GenericFloat},::_LA.RowMaximum=_LA.RowMaximum();check::Bool=true)=LU(_lu_factor!(A;check))

# ==== Cholesky ==================================================================
"""
    OxiBLAS.Cholesky{T,E} <: LinearAlgebra.Factorization{T}

Cholesky factorization A = L·Lᴴ, which wraps an engine of type `E`:

- `CholeskyFactor` (dense, lower triangle);
- `MultifrontalCholeskyFactor` (sparse: `A[p,p] = L·Lᴴ`).

The `info` field is nonzero when the factorization failed (`check=false`).
"""
mutable struct Cholesky{T,E} <: _Fact{T}
    engine::E
    info::Int
end
Cholesky(E::CholeskyFactor{T},info::Integer=0) where T=Cholesky{T,typeof(E)}(E,Int(info))

Base.size(F::Cholesky{<:Any,<:CholeskyFactor})=size(_eng(F).L)
_LA.issuccess(F::Cholesky)=getfield(F,:info)==0
_chol_ready(F)=(getfield(F,:info)==0 || throw(PosDefException(getfield(F,:info))); _eng(F))
function _LA.ldiv!(F::Cholesky{T,<:CholeskyFactor},B::AbstractVecOrMat) where T
    E=_chol_ready(F); _check_rows(F,B)
    _apply_split!(T,B) do X
        _trisolve!(X,E.L,:L,:N,false); _trisolve!(X,E.L,:L,:C,false)
    end
end
function _LA.logabsdet(F::Cholesky{T,<:CholeskyFactor}) where T
    L=_chol_ready(F).L; d=zero(real(T))
    for i in axes(L,1); d+=log(real(L[i,i])); end
    (2d,one(T))
end
function Base.getproperty(F::Cholesky{T,<:CholeskyFactor},s::Symbol) where T
    L=_eng(F).L
    (s===:L || s===:UL) && return _LA.LowerTriangular(L)
    s===:U && return _LA.UpperTriangular(adjoint(L))
    s===:uplo && return 'L'
    s===:factors && return L
    getfield(F,s)
end
Base.propertynames(F::Cholesky{<:Any,<:CholeskyFactor},private::Bool=false)=
    (:L,:U,:UL,(private ? (:uplo,:factors,:info,:engine) : ())...)

# Input reading: the engine reads the lower triangle with getindex. A
# Hermitian/real Symmetric wrapper whose uplo is 'U' supplies its conjugated
# upper triangle that way. Any other input must be Hermitian, or
# PosDefException(-1) is raised, as in LinearAlgebra.
_chol_dense_input(A::Union{_LA.Hermitian{<:Any,<:StridedMatrix},_LA.Symmetric{<:Real,<:StridedMatrix}})=
    eltype(A)<:GenericFloat ? (A.uplo=='L' ? parent(A) : A) : _dense_input(A)
_chol_dense_input(A::_DenseIn)=_LA.ishermitian(A) ? _dense_input(A) : nothing

"""
    OxiBLAS.cholesky(A, [NoPivot()]; check=true) -> OxiBLAS.Cholesky

Cholesky factorization of a Hermitian positive definite matrix
(`cholesky_factor`).

- Accepts `Hermitian`, real `Symmetric` and plain matrices; a plain matrix is
  checked for being Hermitian.
- `check=false` returns a factorization whose `issuccess` is `false` instead
  of throwing `PosDefException`.
- `cholesky!(A)` factors in place: the lower triangle of `A` (a `Matrix` or
  a strided view of floating type) is overwritten by L, its strict upper
  triangle is zeroed, and `A` becomes `F.factors`. For a wrapper with
  `uplo='U'`, the conjugated upper triangle is first mirrored into the lower
  one inside the parent.
"""
function cholesky(A::_DenseIn,::_LA.NoPivot=_LA.NoPivot();check::Bool=true)
    _LA.checksquare(A); X=_chol_dense_input(A)
    T=_la_eltype(eltype(A))
    if X===nothing
        check && throw(PosDefException(-1))
        return Cholesky(CholeskyFactor(Matrix{T}(A)),-1)
    end
    F,info=_cholesky_factor_info(X;check)
    Cholesky(F,info)
end
function _chol_inplace!(A::StridedMatrix,check::Bool)
    F,info=_cholesky_factor_info!(A;check)
    Cholesky(F,info)
end
function cholesky!(A::StridedMatrix{<:GenericFloat},::_LA.NoPivot=_LA.NoPivot();check::Bool=true)
    _LA.checksquare(A)
    if !_LA.ishermitian(A)
        check && throw(PosDefException(-1))
        return Cholesky(CholeskyFactor(A),-1)
    end
    _chol_inplace!(A,check)
end
function cholesky!(H::Union{_LA.Hermitian{<:GenericFloat,<:StridedMatrix},_LA.Symmetric{<:AbstractFloat,<:StridedMatrix}},
                   ::_LA.NoPivot=_LA.NoPivot();check::Bool=true)
    A=parent(H); n=_LA.checksquare(A)
    if H.uplo=='U'
        @inbounds for j in 1:n, i in 1:j-1
            A[j,i]=conj(A[i,j])
        end
    end
    _chol_inplace!(A,check)
end

# ==== self-adjoint factorizations: adjoint and transpose solves =================
# Cholesky is Hermitian: Aᴴ = A. For a Hermitian or real LDLᵀ/Bunch–Kaufman,
# Aᴴ = A; for a complex symmetric one, Aᵀ = A. The remaining case goes through
# conjugation: Aᵀ = conj(A) for a complex Hermitian factorization, and
# Aᴴ = conj(A) for a complex symmetric one. These solves are primitives on
# the Adjoint/TransposeFactorization wrappers, which are kept rather than
# overriding adjoint(F), so that every eltype gets the same wrapper types.
_selfadjoint(F::Cholesky)=true
_selftranspose(F::Cholesky{<:Real})=true
_selftranspose(F::Cholesky)=false

# ==== Bunch–Kaufman and LDLᵀ ====================================================
"""
    OxiBLAS.BunchKaufman{T,E} <: LinearAlgebra.Factorization{T}

Symmetric indefinite factorization with Bunch–Kaufman pivoting (1×1 and 2×2
blocks): `A[p,p] = L·D·Lᴴ`, or `L·D·Lᵀ` for a complex symmetric matrix. It
wraps an engine of type `E`:

- `LDLFactor` (dense);
- `SparseLDLFactor` (sparse, with the MA57 threshold).
"""
mutable struct BunchKaufman{T,E} <: _Fact{T}
    engine::E
end
"""
    OxiBLAS.LDLt{T,E} <: LinearAlgebra.Factorization{T}

LDLᵀ factorization without pivoting: `A[p,p] = L·D·Lᴴ`, with D diagonal and
`p` a fill-reducing ordering (identity for a dense matrix). It wraps an
engine of type `E`:

- `LDLFactor` (dense, `pivot=:none`);
- `SparseLDLFactor` (sparse, `pivot=:none`).
"""
mutable struct LDLt{T,E} <: _Fact{T}
    engine::E
end
const _LDLTypes=Union{BunchKaufman,LDLt}
BunchKaufman(E::Union{LDLFactor{T},SparseLDLFactor{T}}) where T=BunchKaufman{T,typeof(E)}(E)
LDLt(E::Union{LDLFactor{T},SparseLDLFactor{T}}) where T=LDLt{T,typeof(E)}(E)

Base.size(F::_LDLTypes)=(n=length(_eng(F).p);(n,n))
_LA.issuccess(F::_LDLTypes)=true   # the engines throw on breakdown
_selfadjoint(F::_LDLTypes)=eltype(F)<:Real || _eng(F).hermitian
_selftranspose(F::_LDLTypes)=eltype(F)<:Real || !_eng(F).hermitian

function _LA.ldiv!(F::_LDLTypes,B::AbstractVecOrMat)
    E=_eng(F); _check_rows(F,B)
    _apply_split!(X->_ldl_solve!(E,X),eltype(F),B)
end
function _LA.logabsdet(F::Union{BunchKaufman{T,<:LDLFactor},LDLt{T,<:LDLFactor}}) where T
    E=_eng(F); d_vec=getfield(E,:d); dl_vec=getfield(E,:dl); herm=getfield(E,:hermitian)
    d=zero(real(T)); s=one(T); k=1
    for step in E.blocks
        if step==1
            δ=d_vec[k]
        else
            b=dl_vec[k]; bh=herm ? conj(b) : b
            δ=d_vec[k]*d_vec[k+1]-b*bh
        end
        d+=log(abs(δ)); s*=sign(δ); k+=step
    end
    (d,s)
end
function Base.getproperty(F::Union{BunchKaufman{T,<:LDLFactor},LDLt{T,<:LDLFactor}},s::Symbol) where T
    E=_eng(F)
    s===:L && return _LA.UnitLowerTriangular(E.L)
    s===:U && return E.hermitian ? _LA.UnitUpperTriangular(adjoint(E.L)) : _LA.UnitUpperTriangular(transpose(E.L))
    if s===:D
        d_vec=getfield(E,:d); dl_vec=getfield(E,:dl)
        return F isa LDLt ? _LA.Diagonal(copy(d_vec)) : _LA.Tridiagonal(copy(dl_vec),copy(d_vec),E.hermitian ? conj.(dl_vec) : copy(dl_vec))
    end
    s===:p && return E.p
    s===:P && return _permmatrix(T,E.p)
    s===:uplo && return 'L'
    s===:hermitian && return E.hermitian
    getfield(F,s)
end
Base.propertynames(F::Union{BunchKaufman{<:Any,<:LDLFactor},LDLt{<:Any,<:LDLFactor}},private::Bool=false)=
    (:L,:U,:D,:p,:P,(private ? (:uplo,:hermitian,:engine) : ())...)

# Triangle and symmetry kind, read from the wrapper (LinearAlgebra's
# convention for plain matrices: Hermitian first, then symmetric, otherwise
# an ArgumentError).
_sym_kind(A::_LA.Hermitian)=(parent(A),Symbol(A.uplo),true)
_sym_kind(A::_LA.Symmetric)=(parent(A),Symbol(A.uplo),eltype(A)<:Real)
function _sym_kind(A::AbstractMatrix)
    _LA.ishermitian(A) && return (A,:L,true)
    _LA.issymmetric(A) && return (A,:L,false)
    throw(ArgumentError("matrix is neither symmetric nor Hermitian"))
end
function _ldl_dense(A,pivot)
    _LA.checksquare(A); P,uplo,herm=_sym_kind(A)
    ldlt_factor(_dense_input(P);uplo,hermitian=herm,pivot)
end

"""
    OxiBLAS.bunchkaufman(A, rook=false; check=true) -> OxiBLAS.BunchKaufman

Bunch–Kaufman factorization of a symmetric or Hermitian matrix
(`ldlt_factor`, `pivot=:bunchkaufman`).

- Only `rook=false` is available: rook pivoting has no engine.
- A singular matrix throws `SingularException` whatever the value of `check`,
  because the engine reports it while factorizing.
"""
function bunchkaufman(A::Union{StridedMatrix,_DenseHS},rook::Bool=false;check::Bool=true)
    rook && throw(ArgumentError("rook pivoting is not available; use rook=false"))
    BunchKaufman(_ldl_dense(A,:bunchkaufman))
end
bunchkaufman!(A::Union{StridedMatrix,_DenseHS},rook::Bool=false;check::Bool=true)=bunchkaufman(A,rook;check)
"""
    OxiBLAS.ldlt(A) -> OxiBLAS.LDLt

LDLᵀ factorization without pivoting of a symmetric or Hermitian matrix.
LinearAlgebra only provides it for SymTridiagonal (dense) and CHOLMOD
(sparse). It throws on a zero pivot; prefer `bunchkaufman` for indefinite
matrices.
"""
ldlt(A::Union{StridedMatrix,_DenseHS})=LDLt(_ldl_dense(A,:none))

# ==== QR, QRPivoted, LQ =========================================================
"""
    OxiBLAS.QR{T,E} <: LinearAlgebra.Factorization{T}

Householder QR factorization `A = Q·R` (`QRFactor`). `F.Q` is an implicit
`AbstractQ`, and `F \\ b` returns the least-squares solution for a tall `A`
and the minimum-norm solution for a wide one.
"""
struct QR{T,E} <: _Fact{T}
    engine::E
end
QR(E::QRFactor{T}) where T=QR{T,typeof(E)}(E)
"""
    OxiBLAS.QRPivoted{T,E} <: LinearAlgebra.Factorization{T}

QR factorization with column pivoting, `A[:,p] = Q·R` (`PivotedQRFactor`).

- The solution is the minimum-norm one: the numerical rank is read from the
  diagonal of R, with |Rᵢᵢ| > min(m,n)·eps·|R₁₁|.
"""
struct QRPivoted{T,E} <: _Fact{T}
    engine::E
end
QRPivoted(E::PivotedQRFactor{T}) where T=QRPivoted{T,typeof(E)}(E)
"""
    OxiBLAS.LQ{T,E} <: LinearAlgebra.Factorization{T}

LQ factorization `A = L·Q`. It is stored in Householder form as the QR
factorization of Aᴴ, which is what `lq_factor` computes; Q stays implicit.
`F \\ b` returns the minimum-norm solution of a wide system.
"""
struct LQ{T,E} <: _Fact{T}
    engine::E
end
LQ(E::QRFactor{T}) where T=LQ{T,typeof(E)}(E)
"""
    OxiBLAS.QRSparse{T,E} <: LinearAlgebra.Factorization{T}

Multifrontal sparse QR factorization (`SparseQRFactor`) with the SPQR
contract `Q*R == A[prow,pcol]`. The engine folds its row reordering into Q
(transpositions stored as reflectors), so `prow` is the identity. Its
methods are in linalg_api_sparse.jl.
"""
struct QRSparse{T,E} <: _Fact{T}
    engine::E
end

Base.size(F::QR)=size(_eng(F).factors)
Base.size(F::QRPivoted)=size(_eng(F).qr.factors)
Base.size(F::LQ)=reverse(size(_eng(F).factors))
_LA.issuccess(::Union{QR,QRPivoted,LQ})=true

# Solve with E = QRFactor of an m×n matrix A, where B has max(m,n) rows and
# the right-hand side sits in the first m of them.
#   - Rank r = n: least squares through R (square or tall A).
#   - r < n: minimum-norm solution of [R₁₁ R₁₂]·z = (Qᴴb)[1:r], through the
#     QR factorization of the r×n trapezoid, as LinearAlgebra does for wide QR.
function _qr_ls!(E::QRFactor,B::AbstractMatrix,r::Int)
    a=E.factors; m,n=size(a); vec1=size(B,2)==1
    apply_q!(view(B,1:m,:),E;trans=:C)
    if r==n
        _trisolve!(vec1 ? view(B,1:n,1) : view(B,1:n,:),view(a,1:n,1:n),:U,:N,false)
    else
        G=qr_factor(adjoint(_triu_rows(a,r)))
        _trisolve!(vec1 ? view(B,1:r,1) : view(B,1:r,:),view(G.factors,1:r,1:r),:U,:C,false)
        fill!(view(B,r+1:n,:),zero(eltype(B)))
        apply_q!(view(B,1:n,:),G)
    end
    B
end
# Solves Aᴴ·x = b, where Aᴴ is n×m and B has max(m,n) rows.
#   - m ≥ n: minimum norm, x = Q·[R⁻ᴴb; 0].
#   - m < n: least squares through the QR factorization of Rᴴ.
function _qr_adj_ls!(E::QRFactor,B::AbstractMatrix)
    a=E.factors; m,n=size(a); vec1=size(B,2)==1
    if m>=n
        _trisolve!(vec1 ? view(B,1:n,1) : view(B,1:n,:),view(a,1:n,1:n),:U,:C,false)
        fill!(view(B,n+1:m,:),zero(eltype(B)))
        apply_q!(view(B,1:m,:),E)
    else
        G=qr_factor(adjoint(_triu_rows(a,m)))
        apply_q!(view(B,1:n,:),G;trans=:C)
        _trisolve!(vec1 ? view(B,1:m,1) : view(B,1:m,:),view(G.factors,1:m,1:m),:U,:N,false)
        apply_q!(view(B,1:m,:),E)
    end
    B
end
_ldiv_rows(F)=max(size(F)...)
function _LA.ldiv!(F::QR{T},B::AbstractVecOrMat) where T
    _check_rows(F,B,_ldiv_rows(F))
    _apply_split!(X->_qr_ls!(_eng(F),_asmatrix(X),min(size(F)...)),T,B)
end
function _LA.ldiv!(Fa::_AdjF{<:Any,<:QR{T}},B::AbstractVecOrMat) where T
    _check_rows(Fa,B,_ldiv_rows(Fa))
    _apply_split!(X->_qr_adj_ls!(_eng(parent(Fa)),_asmatrix(X)),T,B)
end
function _LA.ldiv!(F::LQ{T},B::AbstractVecOrMat) where T
    _check_rows(F,B,_ldiv_rows(F))
    _apply_split!(X->_qr_adj_ls!(_eng(F),_asmatrix(X)),T,B)
end
function _LA.ldiv!(Fa::_AdjF{<:Any,<:LQ{T}},B::AbstractVecOrMat) where T
    _check_rows(Fa,B,_ldiv_rows(Fa))
    _apply_split!(X->_qr_ls!(_eng(parent(Fa)),_asmatrix(X),min(size(Fa)...)),T,B)
end
function _qr_numerical_rank(a::AbstractMatrix)
    k=min(size(a)...); k==0 && return 0
    tol=k*eps(real(float(one(eltype(a)))))*abs(a[1,1]); r=0
    for i in 1:k
        abs(a[i,i])>tol || break
        r=i
    end
    r
end
function _LA.ldiv!(F::QRPivoted{T},B::AbstractVecOrMat) where T
    _check_rows(F,B,_ldiv_rows(F)); E=_eng(F); n=size(F,2)
    _apply_split!(T,B) do X
        Y=_asmatrix(X)
        _qr_ls!(E.qr,Y,_qr_numerical_rank(E.qr.factors))
        _rowperm!(view(Y,1:n,:),E.p,true)
    end
end

# Specialization (dimensions): the generic `\` (1.10 and 1.14) sizes the
# solution like B, which is wrong when the system is not square. Like
# LinearAlgebra's own QR/LQ `\`, B is padded to max(m,n) rows, solved in
# place, and the first n rows are returned. The second method resolves the
# ambiguity with the generic method for a real F and a complex B.
function _rect_backslash(F,B::AbstractVecOrMat)
    Base.require_one_based_indexing(B)
    m,n=size(F)
    size(B,1)==m || throw(DimensionMismatch("factorization has $m rows, right-hand side has $(size(B,1))"))
    S=typeof(zero(eltype(B))/oneunit(eltype(F)))
    if B isa AbstractVector
        X=m>n ? Vector{S}(undef,m) : zeros(S,max(m,n))
        copyto!(view(X,1:m),B)
        _LA.ldiv!(F,X)
        return m>n ? resize!(X,n) : X
    end
    X=m>n ? Matrix{S}(undef,m,size(B,2)) : zeros(S,max(m,n),size(B,2))
    copyto!(view(X,1:m,:),B)
    _LA.ldiv!(F,X)
    m==n ? X : X[1:n,:]
end
const _RectFact=Union{QR,QRPivoted,LQ,QRSparse,_AdjF{<:Any,<:QR},_AdjF{<:Any,<:LQ}}
Base.:\(F::_RectFact,B::AbstractVecOrMat)=_rect_backslash(F,B)
Base.:\(F::Union{QR{T},QRPivoted{T},LQ{T},QRSparse{T}},B::VecOrMat{Complex{T}}) where {T<:_BlasReal}=_rect_backslash(F,B)
Base.:\(F::Union{_AdjF{T,<:QR},_AdjF{T,<:LQ}},B::VecOrMat{Complex{T}}) where {T<:_BlasReal}=_rect_backslash(F,B)

# ---- implicit Q factors ------------------------------------------------------
"""
    OxiBLAS.QRPackedQ{T,E} <: LinearAlgebra.AbstractQ{T}

The Q factor of a QR factorization, applied through its Householder
reflectors and never formed. `Matrix(Q)` gives the thin m×min(m,n) basis,
as for LinearAlgebra's QR.
"""
struct QRPackedQ{T,E} <: _LA.AbstractQ{T}
    engine::E
end
"""
    OxiBLAS.LQPackedQ{T,E} <: LinearAlgebra.AbstractQ{T}

The Q factor of an LQ factorization, which is the adjoint of the Q factor of
the QR factorization of Aᴴ. `Matrix(Q)` gives the thin min(m,n)×n basis.
"""
struct LQPackedQ{T,E} <: _LA.AbstractQ{T}
    engine::E
end
QRPackedQ(E::QRFactor{T}) where T=QRPackedQ{T,typeof(E)}(E)
LQPackedQ(E::QRFactor{T}) where T=LQPackedQ{T,typeof(E)}(E)
const _PackedQ=Union{QRPackedQ,LQPackedQ}
Base.size(Q::_PackedQ)=(m=size(_eng(Q).factors,1);(m,m))

# apply_q! works on matrices of the factor element type, so the operand goes
# through _apply_split!. For LQ, Q is the adjoint of the stored QR Q, so the
# trans flags are swapped.
_qflag(::QRPackedQ,adj::Bool)=adj ? :C : :N
_qflag(::LQPackedQ,adj::Bool)=adj ? :N : :C
function _qapply!(Q::_PackedQ,B::AbstractVecOrMat,side::Symbol,adj::Bool)
    T=eltype(Q); E=_eng(Q); m=size(Q,1)
    size(B,side===:L ? 1 : 2)==m || throw(DimensionMismatch("Q has size ($m, $m), operand has size $(size(B))"))
    _apply_split!(X->apply_q!(_asmatrix(X),E;side,trans=_qflag(Q,adj)),T,B)
end
_LA.lmul!(Q::_PackedQ,B::AbstractVecOrMat)=_qapply!(Q,B,:L,false)
_LA.lmul!(Q::_AdjQ{<:Any,<:_PackedQ},B::AbstractVecOrMat)=_qapply!(parent(Q),B,:L,true)
_LA.rmul!(B::AbstractMatrix,Q::_PackedQ)=_qapply!(Q,B,:R,false)
_LA.rmul!(B::AbstractMatrix,Q::_AdjQ{<:Any,<:_PackedQ})=_qapply!(parent(Q),B,:R,true)
# Needed for eltype promotion (Q*x with a wider x), as LinearAlgebra's
# QRPackedQ defines it; this copies the reflectors.
_convert_qr(::Type{S},E::QRFactor) where S=QRFactor(Matrix{S}(E.factors),Vector{S}(E.tau))
Base.convert(::Type{_LA.AbstractQ{S}},Q::QRPackedQ{S}) where S=Q
Base.convert(::Type{_LA.AbstractQ{S}},Q::LQPackedQ{S}) where S=Q
Base.convert(::Type{_LA.AbstractQ{S}},Q::QRPackedQ) where S=QRPackedQ(_convert_qr(S,_eng(Q)))
Base.convert(::Type{_LA.AbstractQ{S}},Q::LQPackedQ) where S=LQPackedQ(_convert_qr(S,_eng(Q)))
# Specialization (representation): thin basis, as in LinearAlgebra; the
# generic method would return the full square Q.
Base.Matrix{S}(Q::QRPackedQ) where S=Matrix{S}(qr_q(_eng(Q);columns=min(size(_eng(Q).factors)...)))
Base.Matrix{S}(Q::LQPackedQ) where S=Matrix{S}(adjoint(qr_q(_eng(Q);columns=min(size(_eng(Q).factors)...))))
# det from the number of Householder reflections (LinearAlgebra's _det_tau).
_det_tau(τ::AbstractVector{<:Real})=isodd(count(!iszero,τ)) ? -one(eltype(τ)) : one(eltype(τ))
_det_tau(τ::AbstractVector)=prod(t->iszero(t) ? one(t) : -sign(t)^2,τ;init=one(eltype(τ)))
_LA.det(Q::QRPackedQ)=_det_tau(_eng(Q).tau)
_LA.det(Q::LQPackedQ)=conj(_det_tau(_eng(Q).tau))

function Base.getproperty(F::QR{T},s::Symbol) where T
    E=_eng(F)
    s===:Q && return QRPackedQ(E)
    s===:R && return _triu_rows(E.factors,min(size(E.factors)...))
    s===:factors && return E.factors
    (s===:τ || s===:tau) && return E.tau
    getfield(F,s)
end
Base.propertynames(F::QR,private::Bool=false)=(:R,:Q,(private ? (:factors,:τ,:engine) : ())...)
function Base.getproperty(F::QRPivoted{T},s::Symbol) where T
    E=_eng(F)
    s===:Q && return QRPackedQ(E.qr)
    s===:R && return _triu_rows(E.qr.factors,min(size(E.qr.factors)...))
    s===:p && return E.p
    s===:P && return Matrix(transpose(_permmatrix(T,E.p)))   # A*P == A[:,p]
    s===:factors && return E.qr.factors
    (s===:τ || s===:tau) && return E.qr.tau
    s===:jpvt && return E.p
    getfield(F,s)
end
Base.propertynames(F::QRPivoted,private::Bool=false)=(:R,:Q,:p,:P,(private ? (:factors,:τ,:jpvt,:engine) : ())...)
function Base.getproperty(F::LQ{T},s::Symbol) where T
    E=_eng(F)
    s===:Q && return LQPackedQ(E)
    s===:L && return Matrix(adjoint(_triu_rows(E.factors,min(size(E.factors)...))))
    s===:factors && return E.factors
    (s===:τ || s===:tau) && return E.tau
    getfield(F,s)
end
Base.propertynames(F::LQ,private::Bool=false)=(:L,:Q,(private ? (:factors,:τ,:engine) : ())...)

"""
    OxiBLAS.qr(A, [NoPivot() | ColumnNorm()]) -> OxiBLAS.QR or OxiBLAS.QRPivoted

Householder QR factorization (`qr_factor`). With `ColumnNorm()`, the columns
are pivoted (`qr_pivoted`). `qr!` does not overwrite `A`.
"""
qr(A::_DenseIn,::_LA.NoPivot=_LA.NoPivot())=QR(qr_factor(_dense_input(A)))
qr(A::_DenseIn,::_LA.ColumnNorm)=QRPivoted(qr_pivoted(_dense_input(A)))
qr!(A::StridedMatrix,pivot::Union{_LA.NoPivot,_LA.ColumnNorm}=_LA.NoPivot())=qr(A,pivot)
"""
    OxiBLAS.lq(A) -> OxiBLAS.LQ

LQ factorization, stored as the QR factorization of Aᴴ (`qr_factor`), as
`lq_factor` computes it. `lq!` does not overwrite `A`.
"""
lq(A::_DenseIn)=LQ(qr_factor(adjoint(_dense_input(A))))
lq!(A::StridedMatrix)=lq(A)

# ==== SVD ======================================================================
"""
    OxiBLAS.SVD{T,Tr,E} <: LinearAlgebra.Factorization{T}

Singular value decomposition `A = U·Diagonal(S)·Vt` (`SVDFactor`).

- `F \\ b` returns the minimum-norm least-squares solution. Singular values
  at or below `eps·S[1]` are dropped, as in LinearAlgebra.
- `U, S, V = F` destructures the factorization.
"""
struct SVD{T,Tr,E} <: _Fact{T}
    engine::E
end
SVD(E::SVDFactor{T,R}) where {T,R}=SVD{T,R,typeof(E)}(E)
Base.size(F::SVD)=(size(_eng(F).U,1),size(_eng(F).Vt,2))
_LA.issuccess(::SVD)=true
function _svd_solve!(Y::AbstractVecOrMat{T},E::SVDFactor{T,Tr},B::AbstractVecOrMat{T}) where {T,Tr}
    S=E.S; k=isempty(S) ? 0 : count(s->s>eps(Tr)*S[1],S)
    if Y isa AbstractVector && B isa AbstractVector
        W=Vector{T}(undef,k)
        gemv!(W,view(E.U,:,1:k),B;trans=:C)
        @inbounds for i in 1:k; W[i]/=S[i]; end
        gemv!(Y,view(E.Vt,1:k,:),W;trans=:C)
    else
        Ym=_asmatrix(Y); Bm=_asmatrix(B)
        W=Matrix{T}(undef,k,size(Ym,2))
        gemm!(W,adjoint(view(E.U,:,1:k)),Bm)
        @inbounds for j in axes(W,2), i in 1:k; W[i,j]/=S[i]; end
        gemm!(Ym,adjoint(view(E.Vt,1:k,:)),W)
    end
    Y
end
function _LA.ldiv!(F::SVD{T,Tr},B::AbstractVecOrMat) where {T,Tr}
    _check_rows(F,B,_ldiv_rows(F)); E=_eng(F); m,n=size(F)
    _apply_split!(T,B) do X
        X isa AbstractVector ? _svd_solve!(view(X,1:n),E,view(X,1:m)) : _svd_solve!(view(X,1:n,:),E,view(X,1:m,:))
    end
end
Base.:\(F::SVD,B::AbstractVecOrMat)=_rect_backslash(F,B)
Base.:\(F::SVD{T},B::VecOrMat{Complex{T}}) where {T<:_BlasReal}=_rect_backslash(F,B)
function Base.getproperty(F::SVD,s::Symbol)
    E=_eng(F)
    s===:U && return E.U
    s===:S && return E.S
    s===:Vt && return E.Vt
    s===:V && return adjoint(E.Vt)
    getfield(F,s)
end
Base.propertynames(F::SVD,private::Bool=false)=(:U,:S,:V,:Vt,(private ? (:engine,) : ())...)
Base.iterate(F::SVD,state=1)=state==1 ? (F.U,2) : state==2 ? (F.S,3) : state==3 ? (F.V,4) : nothing

"""
    OxiBLAS.svd(A; full=false) -> OxiBLAS.SVD
    OxiBLAS.svd!(A; full=false) -> OxiBLAS.SVD

Singular value decomposition (`svd_factor`). `svd!` does not overwrite `A`.
On structured matrices (`Bidiagonal`, `Diagonal`), applies specialized algorithms
without full dense conversion.
"""
svd(A::_DenseIn;full::Bool=false)=SVD(svd_factor(_dense_input(A);full))
@doc (@doc svd) svd!
svd!(A::StridedMatrix;full::Bool=false)=svd(A;full)
"""
    OxiBLAS.svdvals(A)
    OxiBLAS.svdvals!(A)

Singular values of `A` in descending order (`singular_values`). On structured matrices
(`Bidiagonal`, `Diagonal`), uses specialized routines without forming dense matrices.
"""
svdvals(A::_DenseIn)=singular_values(_dense_input(A))
@doc (@doc svdvals) svdvals!
svdvals!(A::StridedMatrix)=svdvals(A)
svdvals(F::SVD)=F.S

# ==== eigen ====================================================================
"""
    OxiBLAS.Eigen{T,V,M,W} <: LinearAlgebra.Factorization{T}

Eigendecomposition A·vectors = vectors·Diagonal(values). It shares the
engine's arrays without copying them.

- A Hermitian or real symmetric input goes to `eigen_hermitian` (real values,
  in ascending order).
- Any other input goes to `eigen_general` (complex values).
- `values, vectors = F` destructures it.
- It provides no ldiv!: LinearAlgebra.Eigen has no solve either.
"""
struct Eigen{T,V,M<:AbstractMatrix{T},W<:AbstractVector{V}} <: _Fact{T}
    values::W
    vectors::M
end
Base.size(F::Eigen)=size(F.vectors)
_LA.issuccess(::Eigen)=true
function _LA.logabsdet(F::Eigen)
    λ=F.values; d=zero(real(eltype(λ))); s=one(eltype(λ))
    for x in λ; d+=log(abs(x)); s*=sign(x); end
    (d,s)
end
Base.iterate(F::Eigen,state=1)=state==1 ? (F.values,2) : state==2 ? (F.vectors,3) : nothing

_isherm_input(A::Union{_LA.Hermitian,_LA.Symmetric{<:Real}})=true
_isherm_input(A::AbstractMatrix)=_LA.ishermitian(A)
function _eigen_herm(A;vectors::Bool)
    P,uplo=A isa Union{_LA.Hermitian,_LA.Symmetric} ? (parent(A),Symbol(A.uplo)) : (A,:U)
    eigen_hermitian(_dense_input(P);uplo,vectors)
end
"""
    OxiBLAS.eigen(A) -> OxiBLAS.Eigen

Eigendecomposition of a square matrix: `eigen_hermitian` for Hermitian or
real symmetric input, otherwise `eigen_general`. `eigen!` does not overwrite
`A`.
"""
# LinearAlgebra's default order: real part, then imaginary part.
_eigsortby(λ::Real)=λ
_eigsortby(λ::Complex)=(real(λ),imag(λ))
_eig_sorted(λ,V,sortby)=sortby===nothing ? (λ,V) : (p=sortperm(λ;by=sortby); (λ[p],V[:,p]))
# Balancing (`permute`, `scale`, as in LinearAlgebra) with balance_matrix, then
# back-transformation of the vectors; this is what eigen_general(balance=true)
# does, with the two options kept apart.
function _eigen_balanced(X,permute::Bool,scale::Bool)
    (permute || scale) || return eigen_general(X)
    B=balance_matrix(X;permute,scale); E=eigen_general(B.matrix)
    (;values=E.values,vectors=balance_backtransform(B,E.vectors))
end
function eigen(A::_DenseIn;permute::Bool=true,scale::Bool=true,sortby::Union{Function,Nothing}=_eigsortby)
    _LA.checksquare(A)
    if _isherm_input(A)   # ascending already; sorted only on request
        E=_eigen_herm(A;vectors=true)
        λ,V=sortby===_eigsortby ? (E.values,E.vectors) : _eig_sorted(E.values,E.vectors,sortby)
        return Eigen(λ,V)
    end
    E=_eigen_balanced(_dense_input(A),permute,scale)
    λ,V=_eig_sorted(E.values,E.vectors,sortby); Eigen(λ,V)
end
eigen!(A::Union{StridedMatrix,_DenseHS};kw...)=eigen(A;kw...)
"""
    OxiBLAS.eigvals(A; permute=true, scale=true, sortby=...)
    OxiBLAS.eigvals(A, B)

Eigenvalues of `A` (or generalized eigenvalues of the pencil `A - λ·B`).
Dispatches to structured routines (`SymTridiagonal`, `Diagonal`, `Hermitian`) when applicable.
"""
function eigvals(A::_DenseIn;permute::Bool=true,scale::Bool=true,sortby::Union{Function,Nothing}=_eigsortby)
    _LA.checksquare(A)
    _isherm_input(A) && return _eigen_herm(A;vectors=false).values
    X=_dense_input(A)
    λ=(permute || scale) ? eigenvalues(balance_matrix(X;permute,scale).matrix) : eigenvalues(X)
    sortby===nothing ? λ : sort!(λ;by=sortby)
end
eigvals!(A::Union{StridedMatrix,_DenseHS};kw...)=eigvals(A;kw...)
eigvecs(A::_DenseIn)=eigen(A).vectors
eigvals(F::Eigen)=F.values
eigvecs(F::Eigen)=F.vectors

"""
    OxiBLAS.GeneralizedEigen{T,V,M,W} <: LinearAlgebra.Factorization{T}

Generalized eigendecomposition A·vectors = B·vectors·Diagonal(values). It
shares the engine's arrays without copying them.

- Hermitian/real symmetric wrappers go to `eigen_generalized`: Cholesky of B
  (which must be positive definite), then the Hermitian EVD, with real values
  and B-orthonormal vectors.
- Any other matrices go to `eigen_pencil` (QZ, then the pencil vectors),
  where values = α ./ β, possibly infinite.
- `values, vectors = F` destructures it. It provides no ldiv!, like
  LinearAlgebra.GeneralizedEigen.
"""
struct GeneralizedEigen{T,V,M<:AbstractMatrix{T},W<:AbstractVector{V}} <: _Fact{T}
    values::W
    vectors::M
end
Base.size(F::GeneralizedEigen)=size(F.vectors)
_LA.issuccess(::GeneralizedEigen)=true
Base.iterate(F::GeneralizedEigen,state=1)=state==1 ? (F.values,2) : state==2 ? (F.vectors,3) : nothing
eigvals(F::GeneralizedEigen)=F.values
eigvecs(F::GeneralizedEigen)=F.vectors

const _HermPair=Union{_LA.Hermitian{<:Any,<:StridedMatrix},_LA.Symmetric{<:Real,<:StridedMatrix}}
function _pair_input(A,B)
    X=_dense_input(A); Y=_dense_input(B); S=promote_type(eltype(X),eltype(Y))
    _LA.checksquare(X)==_LA.checksquare(Y) || throw(DimensionMismatch("A has size $(size(A)), B has size $(size(B))"))
    Matrix{S}(X),Matrix{S}(Y)
end
"""
    OxiBLAS.eigen(A, B) -> OxiBLAS.GeneralizedEigen
    OxiBLAS.eigvals(A, B)

Generalized eigenproblem A·x = λ·B·x.

- `Hermitian`/real `Symmetric` wrappers: `eigen_generalized`, which raises
  `PosDefException` if B is not positive definite, as in LinearAlgebra.
- Any other matrices: `eigen_pencil` (QZ).
"""
eigen(A::_HermPair,B::_HermPair)=begin X,Y=_pair_input(A,B); E=eigen_generalized(X,Y); GeneralizedEigen(E.values,E.vectors) end
eigvals(A::_HermPair,B::_HermPair)=begin X,Y=_pair_input(A,B); eigen_generalized(X,Y;vectors=false).values end
eigen(A::_DenseIn,B::_DenseIn)=begin X,Y=_pair_input(A,B); E=eigen_pencil(X,Y); GeneralizedEigen(E.values,E.vectors) end
eigvals(A::_DenseIn,B::_DenseIn)=begin X,Y=_pair_input(A,B); eigen_pencil(X,Y;vectors=false).values end

# ==== Sylvester and Lyapunov ====================================================
"""
    OxiBLAS.sylvester(A, B, C)

Solution X of A·X + X·B + C = 0, LinearAlgebra's convention. Bartels–Stewart
on the native engines:

1. Schur forms A = U·S·Uᴴ and B = V·T·Vᴴ (`schur_factor`).
2. S·Y + Y·T = -Uᴴ·C·V, solved by `generalized_sylvester` (TGSYL) with the
   pencils (S, I) and (-T, I) and a zero second right-hand side, which forces
   R = L.
3. X = U·Y·Vᴴ.

The equation has a unique solution only when no λᵢ(A) + μⱼ(B) is zero.
"""
function sylvester(A::_DenseIn,B::_DenseIn,C::_DenseIn)
    S=promote_type(_la_eltype(eltype(A)),_la_eltype(eltype(B)),_la_eltype(eltype(C)))
    p=_LA.checksquare(A); q=_LA.checksquare(B)
    size(C)==(p,q) || throw(DimensionMismatch("C has size $(size(C)), expected ($p, $q)"))
    FA=schur_factor(Matrix{S}(A)); FB=schur_factor(Matrix{S}(B))
    Z1=Matrix{S}(FA.Z); Z2=Matrix{S}(FB.Z); TA=Matrix{S}(FA.T); TB=Matrix{S}(FB.T)
    Ct=_mm(_mm(Matrix{S}(adjoint(Z1)),Matrix{S}(C)),Z2)
    sol=generalized_sylvester(TA,_identity(S,p),-TB,_identity(S,q),-Ct,zeros(S,p,q))
    Y=sol.R ./ sol.scale
    _mm(_mm(Z1,Y),Matrix{S}(adjoint(Z2)))
end
"""
    OxiBLAS.lyap(A, C)

Solution X of A·X + X·Aᴴ + C = 0 (continuous Lyapunov equation), as
`sylvester(A, Aᴴ, C)`.
"""
lyap(A::_DenseIn,C::_DenseIn)=sylvester(A,Matrix(adjoint(A)),C)

# ==== Schur, generalized Schur, Hessenberg ======================================
"""
    OxiBLAS.Schur{T,C,E} <: LinearAlgebra.Factorization{T}

Schur decomposition `A = Z·T·Zᴴ` (`SchurFactor`). For real input, T is real
quasi-triangular. It has the properties `T`, `Z` (also `vectors`) and
`values`, and destructures as `T, Z, values = F`.
"""
mutable struct Schur{T,C,E} <: _Fact{T}   # mutable: ordschur! replaces the factors in F
    engine::E
end
Schur(E::SchurFactor{T,C}) where {T,C}=Schur{T,C,typeof(E)}(E)
Base.size(F::Schur)=size(_eng(F).T)
function Base.getproperty(F::Schur,s::Symbol)
    E=_eng(F)
    s===:T && return E.T
    (s===:Z || s===:vectors) && return E.Z
    s===:values && return E.values
    getfield(F,s)
end
Base.propertynames(F::Schur,private::Bool=false)=(:T,:Z,:vectors,:values,(private ? (:engine,) : ())...)
Base.iterate(F::Schur,state=1)=state==1 ? (F.T,2) : state==2 ? (F.Z,3) : state==3 ? (F.values,4) : nothing
"""
    OxiBLAS.GeneralizedSchur{T,C,E} <: LinearAlgebra.Factorization{T}

Generalized Schur (QZ) decomposition `A = Q·S·Zᴴ`, `B = Q·T·Zᴴ`
(`QZFactor`). It has the properties `S`, `T`, `Q`, `Z`, `α`, `β` and
`values = α ./ β`.
"""
mutable struct GeneralizedSchur{T,C,E} <: _Fact{T}
    engine::E
end
GeneralizedSchur(E::QZFactor{T,C}) where {T,C}=GeneralizedSchur{T,C,typeof(E)}(E)
Base.size(F::GeneralizedSchur)=size(_eng(F).S)
function Base.getproperty(F::GeneralizedSchur,s::Symbol)
    E=_eng(F)
    s===:S && return E.S
    s===:T && return E.T
    (s===:Q || s===:left) && return E.Q
    (s===:Z || s===:right) && return E.Z
    (s===:α || s===:alpha) && return E.alpha
    (s===:β || s===:beta) && return E.beta
    s===:values && return E.alpha ./ E.beta
    getfield(F,s)
end
Base.propertynames(F::GeneralizedSchur,private::Bool=false)=(:S,:T,:Q,:Z,:α,:β,:values,(private ? (:left,:right,:engine) : ())...)
Base.iterate(F::GeneralizedSchur,state=1)=state<=6 ? ((F.S,F.T,F.α,F.β,F.Q,F.Z)[state],state+1) : nothing
"""
    OxiBLAS.Hessenberg{T,E} <: LinearAlgebra.Factorization{T}

Hessenberg reduction `A = Q·H·Qᴴ` (`HessenbergFactor`).

- `F.H` is an `UpperHessenberg` matrix.
- `F.Q` is an implicit `HessenbergQ`.
- It provides no ldiv!, since the engine has no shifted Hessenberg solve.
"""
struct Hessenberg{T,E} <: _Fact{T}
    engine::E
end
Hessenberg(E::HessenbergFactor{T}) where T=Hessenberg{T,typeof(E)}(E)
Base.size(F::Hessenberg)=size(_eng(F).factors)
"""
    OxiBLAS.HessenbergQ{T,E} <: LinearAlgebra.AbstractQ{T}

The Q factor of a Hessenberg reduction, applied through its reflectors
(`apply_hessenberg_q!`).
"""
struct HessenbergQ{T,E} <: _LA.AbstractQ{T}
    engine::E
end
HessenbergQ(E::HessenbergFactor{T}) where T=HessenbergQ{T,typeof(E)}(E)
Base.size(Q::HessenbergQ)=size(_eng(Q).factors)
function _hqapply!(Q::HessenbergQ,B::AbstractVecOrMat,side::Symbol,adj::Bool)
    m=size(Q,1)
    size(B,side===:L ? 1 : 2)==m || throw(DimensionMismatch("Q has size ($m, $m), operand has size $(size(B))"))
    _apply_split!(X->apply_hessenberg_q!(X,_eng(Q);side,trans=adj ? :C : :N),eltype(Q),B)
end
_LA.lmul!(Q::HessenbergQ,B::AbstractVecOrMat)=_hqapply!(Q,B,:L,false)
_LA.lmul!(Q::_AdjQ{<:Any,<:HessenbergQ},B::AbstractVecOrMat)=_hqapply!(parent(Q),B,:L,true)
_LA.rmul!(B::AbstractMatrix,Q::HessenbergQ)=_hqapply!(Q,B,:R,false)
_LA.rmul!(B::AbstractMatrix,Q::_AdjQ{<:Any,<:HessenbergQ})=_hqapply!(parent(Q),B,:R,true)
Base.convert(::Type{_LA.AbstractQ{S}},Q::HessenbergQ{S}) where S=Q
Base.convert(::Type{_LA.AbstractQ{S}},Q::HessenbergQ) where S=
    HessenbergQ(HessenbergFactor(Matrix{S}(_eng(Q).factors),_eng(Q).tau,_eng(Q).scale))
_LA.det(Q::HessenbergQ)=_det_tau(_eng(Q).tau)
function Base.getproperty(F::Hessenberg,s::Symbol)
    E=_eng(F)
    s===:H && return _LA.UpperHessenberg(_hessenberg_matrix(E))
    s===:Q && return HessenbergQ(E)
    getfield(F,s)
end
Base.propertynames(F::Hessenberg,private::Bool=false)=(:H,:Q,(private ? (:engine,) : ())...)
Base.iterate(F::Hessenberg,state=1)=state==1 ? (F.Q,2) : state==2 ? (F.H,3) : nothing

"""
    OxiBLAS.schur(A) -> OxiBLAS.Schur
    OxiBLAS.schur(A, B) -> OxiBLAS.GeneralizedSchur

Schur decomposition (`schur_factor`), or generalized Schur decomposition by
QZ (`qz_factor`). `schur!` does not overwrite its arguments.
"""
schur(A::_DenseIn)=Schur(schur_factor(_dense_input(A)))
schur(A::_DenseIn,B::_DenseIn)=(X=_dense_input(A);Y=_dense_input(B);
    S=promote_type(eltype(X),eltype(Y)); GeneralizedSchur(qz_factor(Matrix{S}(X),Matrix{S}(Y))))
@doc (@doc schur) schur!
schur!(A::StridedMatrix)=schur(A)
schur!(A::StridedMatrix,B::StridedMatrix)=schur(A,B)
"""
    OxiBLAS.hessenberg(A) -> OxiBLAS.Hessenberg
    OxiBLAS.hessenberg!(A) -> OxiBLAS.Hessenberg

Hessenberg reduction (`hessenberg_factor`). `hessenberg!` does not overwrite
`A`.
"""
hessenberg(A::_DenseIn)=Hessenberg(hessenberg_factor(_dense_input(A)))
@doc (@doc hessenberg) hessenberg!
hessenberg!(A::StridedMatrix)=hessenberg(A)

# ==== solves shared by the square factorizations ================================
const _SelfAdj=Union{Cholesky,BunchKaufman,LDLt}
# A Hermitian or real factorization is its own adjoint; a complex symmetric
# one gets conj(F \ conj(B)). The same holds with the roles of adjoint and
# transpose swapped.
function _LA.ldiv!(Fw::_AdjF{<:Any,<:_SelfAdj},B::AbstractVecOrMat)
    F=parent(Fw)
    _selfadjoint(F) ? _LA.ldiv!(F,B) : conj!(_LA.ldiv!(F,conj!(B)))
end
function _LA.ldiv!(Fw::_TrF{<:Any,<:_SelfAdj},B::AbstractVecOrMat)
    F=parent(Fw)
    _selftranspose(F) ? _LA.ldiv!(F,B) : conj!(_LA.ldiv!(F,conj!(B)))
end

# rdiv!(B, F) for a plain F has no generic method in LinearAlgebra; its own
# LU defines it the same way: B/A = (Aᵀ \ Bᵀ)ᵀ. A copy is needed because the
# kernels do not take lazy Transpose operands.
const _SquareFact=Union{LU,Cholesky,BunchKaufman,LDLt}
function _LA.rdiv!(B::AbstractMatrix,F::_SquareFact)
    X=copy(transpose(B)); _LA.ldiv!(transpose(F),X); copyto!(B,transpose(X)); B
end

# ==== shared queries and display ================================================
const _OxiFact=Union{LU,Cholesky,BunchKaufman,LDLt,QR,QRPivoted,LQ,QRSparse,SVD,Eigen,GeneralizedEigen,Schur,GeneralizedSchur,Hessenberg}
Base.size(F::_OxiFact,d::Integer)=d<1 ? throw(ArgumentError("dimension must be ≥ 1, got $d")) : d<=2 ? size(F)[d] : 1
function Base.show(io::IO,::MIME"text/plain",F::_OxiFact)
    summary(io,F); print(io," of size ",size(F,1),"×",size(F,2))
    if F isa Union{LU,Cholesky}
        print(io,_LA.issuccess(F) ? "" : " (failed)")
    end
end
issuccess(F::_OxiFact)=_LA.issuccess(F)

# ==== functions on matrices =====================================================
"""
    OxiBLAS.factorize(A)

Chooses a factorization from the structure of `A`:

- square, Hermitian: `Cholesky` if it is positive definite, otherwise
  `BunchKaufman`;
- other square matrices: `LU`;
- rectangular: `QRPivoted`.

Triangular and diagonal structures are not detected. Unlike
`LinearAlgebra.factorize`, the result is never a LinearAlgebra wrapper,
whose solves would go through BLAS.
"""
function _has_pos_real_diag(A::AbstractMatrix)
    @inbounds for i in axes(A,1)
        d=A[i,i]
        (isreal(d) && isfinite(real(d)) && real(d)>0) || return false
    end
    true
end
function factorize(A::_DenseIn)
    m,n=size(A)
    m==n || return qr(A,_LA.ColumnNorm())
    if _isherm_input(A)
        if _has_pos_real_diag(A)
            C=cholesky(A;check=false)
            _LA.issuccess(C) && return C
        end
        return bunchkaufman(A isa _DenseHS ? A : _LA.Hermitian(A isa StridedMatrix ? A : _dense_input(A),:L))
    end
    lu(A)
end
factorize(F::_OxiFact)=F

"""
    OxiBLAS.det(A)
    OxiBLAS.logdet(A)
    OxiBLAS.logabsdet(A)

Determinant and log-determinant functions.
- Dense matrices: computed via LU factorization.
- Structured matrices (`Tridiagonal`, `SymTridiagonal`): computed via zero-allocation
  Usmani recursion (`O(n)` time, `O(1)` memory).
- `Diagonal`, `Bidiagonal`, and triangular matrices: computed directly along the diagonal.
- On factorizations: forwarded to `LinearAlgebra`'s generic methods using `logabsdet`.
"""
det(A::_DenseIn)=_LA.det(lu(A;check=false))
@doc (@doc det) logdet
logdet(A::_DenseIn)=_LA.logdet(lu(A;check=false))
@doc (@doc det) logabsdet
logabsdet(A::_DenseIn)=_LA.logabsdet(lu(A;check=false))
det(F::_OxiFact)=_LA.det(F)
logdet(F::_OxiFact)=_LA.logdet(F)
logabsdet(F::_OxiFact)=_LA.logabsdet(F)

"""
    OxiBLAS.inv(A)

Inverse of a square matrix (`native_inverse`). On an OxiBLAS factorization,
it uses LinearAlgebra's generic `inv`, built on `ldiv!`. Sparse matrices are
rejected, as in SparseArrays.
"""
inv(A::_DenseIn)=native_inverse(_dense_input(A))
inv(F::_OxiFact)=Base.inv(F)

# Default tolerances follow LinearAlgebra: rtol = min(m,n)·eps when atol is
# zero.
_la_rtol(A,atol)=iszero(atol) ? min(size(A)...)*eps(real(float(one(eltype(A))))) : zero(real(float(one(eltype(A)))))
"""
    OxiBLAS.rank(A; atol=0, rtol=...)

Numerical rank of `A` from its singular values (`numerical_rank`).
"""
rank(A::_DenseIn;atol::Real=0,rtol::Real=_la_rtol(A,atol))=numerical_rank(_dense_input(A);atol,rtol)
"""
    OxiBLAS.pinv(A; atol=0, rtol=...)

Moore–Penrose pseudoinverse of `A` (`pseudoinverse`).
"""
pinv(A::_DenseIn;atol::Real=0,rtol::Real=_la_rtol(A,atol))=pseudoinverse(_dense_input(A);atol,rtol)
"""
    OxiBLAS.nullspace(A; atol=0, rtol=...)

Orthonormal basis of the null space of `A` (`null_space`).
"""
nullspace(A::_DenseIn;atol::Real=0,rtol::Real=_la_rtol(A,atol))=null_space(_dense_input(A);atol,rtol)
"""
    OxiBLAS.cond(A, p=2)

Condition number of `A` in the 1-, 2- or ∞-norm (`condition_number`).
"""
function cond(A::_DenseIn,p::Real=2)
    p in (1,2,Inf) || throw(ArgumentError("p must be 1, 2 or Inf"))
    condition_number(_dense_input(A);norm=p==2 ? :two : p==1 ? 1 : Inf)
end
"""
    OxiBLAS.isposdef(A)

Whether `A` is Hermitian positive definite, decided by an attempted Cholesky
factorization.
"""
isposdef(A::_DenseIn)=size(A,1)==size(A,2) && _isherm_input(A) && _has_pos_real_diag(A) && _LA.issuccess(cholesky(A;check=false))

# ==== ordschur, norms, kron, extreme eigenvalues ================================
# LinearAlgebra selects by position (in the order of F.values); schur_reorder
# and qz_reorder select by eigenvalue. A repeated eigenvalue that is both
# selected and unselected cannot be separated, by the engine or by LAPACK,
# so it is rejected.
function _value_selector(values::AbstractVector,select::AbstractVector{Bool})
    length(select)==length(values) || throw(DimensionMismatch("select has length $(length(select)), expected $(length(values))"))
    chosen=values[select]; rest=values[.!select]
    any(in(chosen),rest) && throw(ArgumentError("a repeated eigenvalue is both selected and unselected and cannot be separated"))
    λ->λ in chosen
end
"""
    OxiBLAS.ordschur(F::OxiBLAS.Schur, select) / ordschur!(F, select)
    OxiBLAS.ordschur(F::OxiBLAS.GeneralizedSchur, select) / ordschur!(F, select)

Reorders the (generalized) Schur form so that the selected eigenvalues come
first: `schur_reorder` (unitary swaps), or `qz_reorder` for QZ.

- `select` is a Bool vector in the order of `F.values`, as in LinearAlgebra.
  Selecting one value of a real conjugate pair selects the whole 2×2 block.
- `ordschur!` replaces the factors inside `F` and returns `F`; the original
  arrays are not overwritten.
"""
ordschur(F::Schur,select::AbstractVector{Bool})=Schur(schur_reorder(_eng(F),_value_selector(_eng(F).values,select)))
@doc (@doc ordschur) ordschur!
ordschur!(F::Schur,select::AbstractVector{Bool})=(setfield!(F,:engine,schur_reorder(_eng(F),_value_selector(_eng(F).values,select))); F)
ordschur(F::GeneralizedSchur,select::AbstractVector{Bool})=GeneralizedSchur(qz_reorder(_eng(F),_value_selector(_eng(F).values,select)))
ordschur!(F::GeneralizedSchur,select::AbstractVector{Bool})=(setfield!(F,:engine,qz_reorder(_eng(F),_value_selector(_eng(F).values,select))); F)
_LA.ordschur(F::Union{Schur,GeneralizedSchur},select::Union{Vector{Bool},BitVector})=ordschur(F,select)
_LA.ordschur!(F::Union{Schur,GeneralizedSchur},select::Union{Vector{Bool},BitVector})=ordschur!(F,select)

_float_vec(x::StridedVector{<:GenericFloat})=x
_float_vec(x::AbstractVector)=Vector{_la_eltype(eltype(x))}(x)
"""
    OxiBLAS.norm(A, p=2)

Norm of the entries, as in LinearAlgebra.

- On a matrix: `p=2` is the Frobenius norm (`matrix_norm(:fro)`), `1` the
  sum of moduli, `Inf` the largest modulus.
- On a vector: `p=2` uses `nrm2`.

Other `p` values use the generic sum (sum |x|ᵖ)^(1/p).
"""
function norm(A::_DenseIn,p::Real=2)
    X=_dense_input(A); R=real(eltype(X))
    p==2 && return matrix_norm(X,:fro)
    p==1 && return sum(abs,X;init=zero(R))
    p==Inf && return maximum(abs,X;init=zero(R))
    p==-Inf && return minimum(abs,X;init=R(Inf))
    p==0 && return R(count(!iszero,X))
    sum(x->abs(x)^p,X;init=zero(R))^inv(p)
end
function norm(x::AbstractVector,p::Real=2)
    v=_float_vec(x); R=real(eltype(v))
    p==2 && return v isa StridedVector ? nrm2(v) : sqrt(sum(abs2,v;init=zero(R)))
    p==1 && return sum(abs,v;init=zero(R))
    p==Inf && return maximum(abs,v;init=zero(R))
    p==-Inf && return minimum(abs,v;init=R(Inf))
    p==0 && return R(count(!iszero,v))
    sum(t->abs(t)^p,v;init=zero(R))^inv(p)
end
"""
    OxiBLAS.opnorm(A, p=2)

Operator norm (`matrix_norm`): 1 (column sums), 2 (largest singular value)
or Inf (row sums).
"""
function opnorm(A::_DenseIn,p::Real=2)
    p in (1,2,Inf) || throw(ArgumentError("p must be 1, 2 or Inf"))
    matrix_norm(_dense_input(A),p==2 ? :two : p)
end
"""
    OxiBLAS.kron(A, B)

Kronecker product (`kron_product`), with the element types promoted.
"""
function kron(A::_DenseIn,B::_DenseIn)
    S=promote_type(_la_eltype(eltype(A)),_la_eltype(eltype(B)))
    kron_product(Matrix{S}(A),Matrix{S}(B))
end
"""
    OxiBLAS.eigmax(A), OxiBLAS.eigmin(A)

Largest or smallest eigenvalue, from `OxiBLAS.eigvals`, which picks the
structured engine when there is one. The engines may return complex values
whose imaginary parts are exactly zero; those are compared by their real
parts. A `DomainError` is raised only when some eigenvalue really is complex
(nonzero imaginary part), which is LinearAlgebra's contract.
"""
eigmax
@doc (@doc eigmax) eigmin
for (f,g) in ((:eigmax,:maximum),(:eigmin,:minimum))
    @eval function $f(A::AbstractMatrix)
        v=eigvals(A)
        if eltype(v)<:Complex
            all(x->iszero(imag(x)),v) || throw(DomainError(A,$(string(f))*" is only defined for matrices with real eigenvalues"))
            return $g(real,v)
        end
        $g(v)
    end
end

# Matrix functions (exp/log/sqrt/sin/cos): linalg_api_structured.jl, which
# knows every structure whose result keeps its form.
