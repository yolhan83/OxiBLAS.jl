# SPDX-License-Identifier: Apache-2.0
# High-level API for sparse matrices (see the header of linalg_api.jl).
#
# - Accepted inputs are SparseMatrixCSC and the OxiBLAS formats (_OxiSparse),
#   plus their transposes, adjoints and Hermitian/Symmetric wrappers.
# - The multifrontal and supernodal engines are used as they are; no input is
#   ever densified.
# - LinearAlgebra functions are extended only on the OxiBLAS formats, never
#   on SparseMatrixCSC: that one belongs to SparseArrays and SuiteSparse.

const _SpIn=Union{SparseMatrixCSC,_OxiSparse}
const _SpAT=Union{Transpose{<:Any,<:_SpIn},Adjoint{<:Any,<:_SpIn}}
const _SpHerm=Union{_LA.Hermitian{<:Any,<:_SpIn},_LA.Symmetric{<:Real,<:_SpIn}}
const _OxiHerm=Union{_LA.Hermitian{<:Any,<:_OxiSparse},_LA.Symmetric{<:Real,<:_OxiSparse}}

_sparse_input(A::_SpIn)=eltype(A)<:GenericFloat ? A : _mul_convert(_la_eltype(eltype(A)),A)
# Hermitian wrapper -> full Hermitian matrix with both triangles (needed by
# LU and QR, which read the whole pattern).
_sparse_full(A::_SpHerm)=_sparse_hermitian(_sparse_input(parent(A)),Symbol(A.uplo))
# Triangle and symmetry kind of a sparse operand.
_sp_kind(A::_LA.Hermitian{<:Any,<:_SpIn})=(_sparse_input(parent(A)),Symbol(A.uplo),true)
_sp_kind(A::_LA.Symmetric{<:Any,<:_SpIn})=(_sparse_input(parent(A)),Symbol(A.uplo),eltype(A)<:Real)
function _sp_kind(A::_SpIn)
    _LA.ishermitian(A) && return (_sparse_input(A),:L,true)
    _LA.issymmetric(A) && return (_sparse_input(A),:L,false)
    throw(ArgumentError("matrix is neither symmetric nor Hermitian"))
end

# ==== LU ========================================================================
LU(E::MultifrontalLUFactor{T}) where T=LU{T,typeof(E)}(E)
Base.size(F::LU{<:Any,<:MultifrontalLUFactor})=(n=length(_eng(F).p);(n,n))
_LA.issuccess(F::LU{<:Any,<:MultifrontalLUFactor})=(E=_eng(F); all(i->!iszero(E.U[i,i]),eachindex(E.p)))
for (W,tr) in ((:LU,:(:N)),(:_AdjF,:(:C)),(:_TrF,:(:T)))
    S=W===:LU ? :(LU{T,<:MultifrontalLUFactor}) : :($W{<:Any,<:LU{T,<:MultifrontalLUFactor}})
    @eval function _LA.ldiv!(Fw::$S,B::AbstractVecOrMat) where T
        E=_eng($(W===:LU ? :Fw : :(parent(Fw)))); _check_rows(Fw,B)
        _apply_split!(X->_lu_ldiv!(E,X;trans=$tr),T,B)
    end
end
_LA.logabsdet(F::LU{<:Any,<:MultifrontalLUFactor})=logabsdet_factor(_eng(F))
function Base.getproperty(F::LU{<:Any,<:MultifrontalLUFactor},s::Symbol)
    E=_eng(F)
    s===:L && return SparseMatrixCSC(E.L)
    s===:U && return SparseMatrixCSC(E.U)
    s===:p && return E.p
    s===:q && return E.q
    getfield(F,s)
end
Base.propertynames(F::LU{<:Any,<:MultifrontalLUFactor},private::Bool=false)=(:L,:U,:p,:q,(private ? (:engine,) : ())...)
SparseArrays.nnz(F::LU{<:Any,<:MultifrontalLUFactor})=length(_eng(F).L.values)+length(_eng(F).U.values)

"""
    OxiBLAS.lu(A::SparseMatrixCSC / CSRMatrix / ...; check=true) -> OxiBLAS.LU

Sparse LU factorization with threshold pivoting (`multifrontal_lu`,
`ordering=:auto`), giving `F.L*F.U == A[F.p,F.q]`. With `check=false`, a
singular matrix returns a factor with a zero diagonal entry in `U` and
`issuccess(F) == false`.
"""
lu(A::_SpIn;check::Bool=true)=LU(multifrontal_lu(_sparse_input(A);check))
lu(A::_SpAT;check::Bool=true)=lu(copy(A);check)
lu(A::_SpHerm;check::Bool=true)=lu(_sparse_full(A);check)
"""
    OxiBLAS.lu!(F::OxiBLAS.LU, A; check=true) -> F

Refactorizes `A` into `F`. The engine has no separate symbolic phase to
reuse, so the analysis is redone.
"""
function lu!(F::LU{T,<:MultifrontalLUFactor},A::_SpIn;check::Bool=true) where T
    size(A)==size(F) || throw(DimensionMismatch("new matrix has size $(size(A)), factorization has size $(size(F))"))
    setfield!(F,:engine,multifrontal_lu(_mul_convert(T,A);check)); F
end

# ==== Cholesky ==================================================================
Cholesky(E::MultifrontalCholeskyFactor{T},info::Integer=0) where T=Cholesky{T,typeof(E)}(E,Int(info))
Base.size(F::Cholesky{<:Any,<:MultifrontalCholeskyFactor})=(n=length(_eng(F).p);(n,n))
function _LA.ldiv!(F::Cholesky{T,<:MultifrontalCholeskyFactor},B::AbstractVecOrMat) where T
    E=_chol_ready(F); _check_rows(F,B)
    _apply_split!(X->_chol_ldiv!(E,X),T,B)
end
function _LA.logabsdet(F::Cholesky{T,<:MultifrontalCholeskyFactor}) where T
    L=_chol_ready(F).L; d=zero(real(T))
    @inbounds for j in axes(L,2)
        v=zero(T)
        for q in L.colptr[j]:L.colptr[j+1]-1
            L.rowind[q]==j && (v=L.values[q]; break)
        end
        d+=log(real(v))
    end
    (2d,one(T))
end
function Base.getproperty(F::Cholesky{<:Any,<:MultifrontalCholeskyFactor},s::Symbol)
    E=_eng(F)
    s===:L && return SparseMatrixCSC(E.L)
    s===:U && return copy(adjoint(SparseMatrixCSC(E.L)))
    s===:p && return E.p
    s===:PtL && return SparseMatrixCSC(E.L)[invperm(E.p),:]   # A == PtL*PtL'
    getfield(F,s)
end
Base.propertynames(F::Cholesky{<:Any,<:MultifrontalCholeskyFactor},private::Bool=false)=
    (:L,:U,:p,:PtL,(private ? (:info,:engine) : ())...)
SparseArrays.nnz(F::Cholesky{<:Any,<:MultifrontalCholeskyFactor})=length(_eng(F).L.values)

function _sp_chol_operand(A)
    A isa _SpHerm && return (_sparse_input(parent(A)),Symbol(A.uplo))
    _LA.ishermitian(A) || throw(PosDefException(-1))
    (_sparse_input(A),:L)
end
"""
    OxiBLAS.cholesky(A::sparse; check=true, perm=nothing, shift=0) -> OxiBLAS.Cholesky

Supernodal multifrontal Cholesky factorization (`multifrontal_cholesky`) with
an AMD ordering, or the user-supplied permutation vector `perm`; factors
`(A + shift*I)[p,p] = L·Lᴴ` (like CHOLMOD's `perm` and `shift` keywords).

- Accepts a `Hermitian` or real `Symmetric` wrapper (the triangle named by
  its `uplo`), or a plain matrix, which is checked for being Hermitian.
- `perm`: custom fill-reducing permutation vector of `1:n` (`nothing` uses AMD).
- `shift`: real diagonal shift `σ` added to `A` before factoring (`A + σI`).
- A matrix that is not positive definite throws `PosDefException` whatever
  the value of `check`: the engine has no failure state to return.
"""
# shift=σ factors A+σI, as CHOLMOD does: σI is added in OxiBLAS sparse format
# (sparse_add), so the pattern only gains the diagonal.
function _sp_shift(X,shift::Number)
    iszero(shift) && return X
    n=size(X,1); T=eltype(X)
    sparse_add(X,CSRMatrix(n,n,collect(1:n+1),collect(1:n),ones(T,n)),one(T),T(shift))
end
function cholesky(A::Union{_SpIn,_SpHerm};check::Bool=true,perm::Union{Nothing,AbstractVector{<:Integer}}=nothing,shift::Number=0)
    _LA.checksquare(A); X,uplo=_sp_chol_operand(A)
    Cholesky(multifrontal_cholesky(_sp_shift(X,shift);uplo,ordering=perm===nothing ? :amd : perm))
end
cholesky(A::_SpAT;kw...)=cholesky(copy(A);kw...)
"""
    OxiBLAS.cholesky!(F::OxiBLAS.Cholesky, A; check=true, shift=0) -> F

Refactorizes `A + shift*I`, which has the same pattern, into `F`. The ordering
`F.p` is reused, so the AMD step is skipped; the rest of the symbolic analysis
is redone.
"""
function cholesky!(F::Cholesky{T,<:MultifrontalCholeskyFactor},A::Union{_SpIn,_SpHerm};check::Bool=true,shift::Number=0) where T
    size(A)==size(F) || throw(DimensionMismatch("new matrix has size $(size(A)), factorization has size $(size(F))"))
    X,uplo=_sp_chol_operand(A)
    setfield!(F,:engine,multifrontal_cholesky(_sp_shift(_mul_convert(T,X),shift);uplo,ordering=_eng(F).p)); setfield!(F,:info,0); F
end

# ==== LDLᵀ and Bunch–Kaufman ====================================================
function _LA.issuccess(F::Union{BunchKaufman{<:Any,<:SparseLDLFactor},LDLt{<:Any,<:SparseLDLFactor}})
    E=_eng(F); k=1
    @inbounds for step in E.blocks
        step==1 && iszero(E.diagonal[k]) && return false
        k+=step
    end
    true
end
_LA.logabsdet(F::Union{BunchKaufman{<:Any,<:SparseLDLFactor},LDLt{<:Any,<:SparseLDLFactor}})=logabsdet_factor(_eng(F))
function _sp_ldl_tridiagonal(E::SparseLDLFactor{T}) where T
    n=length(E.diagonal); dl=zeros(T,max(n-1,0)); du=zeros(T,max(n-1,0)); k=1
    for step in E.blocks
        if step==2
            dl[k]=E.offdiagonal[k]; du[k]=E.hermitian ? conj(E.offdiagonal[k]) : E.offdiagonal[k]
        end
        k+=step
    end
    _LA.Tridiagonal(dl,copy(E.diagonal),du)
end
function Base.getproperty(F::Union{BunchKaufman{<:Any,<:SparseLDLFactor},LDLt{<:Any,<:SparseLDLFactor}},s::Symbol)
    E=_eng(F)
    s===:L && return SparseMatrixCSC(E.L)
    s===:D && return F isa LDLt ? _LA.Diagonal(copy(E.diagonal)) : _sp_ldl_tridiagonal(E)
    s===:p && return E.p
    s===:hermitian && return E.hermitian
    getfield(F,s)
end
Base.propertynames(F::Union{BunchKaufman{<:Any,<:SparseLDLFactor},LDLt{<:Any,<:SparseLDLFactor}},private::Bool=false)=
    (:L,:D,:p,(private ? (:hermitian,:engine) : ())...)
SparseArrays.nnz(F::Union{BunchKaufman{<:Any,<:SparseLDLFactor},LDLt{<:Any,<:SparseLDLFactor}})=length(_eng(F).L.values)

function _sp_ldl(A,pivot,perm,shift::Number=0;check::Bool=true)
    _LA.checksquare(A); X,uplo,herm=_sp_kind(A)
    sparse_ldlt(_sp_shift(X,shift);uplo,hermitian=herm,pivot,ordering=perm===nothing ? :amd : perm,check)
end
const _SpSym=Union{_SpIn,_LA.Hermitian{<:Any,<:_SpIn},_LA.Symmetric{<:Any,<:_SpIn}}
"""
    OxiBLAS.ldlt(A::sparse; check=true, perm=nothing, shift=0) -> OxiBLAS.LDLt

Multifrontal LDLᵀ factorization without pivoting (`sparse_ldlt`,
`pivot=:none`), with an AMD ordering or the permutation `perm` and optional
diagonal shift `shift` (`A + shift*I`), as in CHOLMOD's `ldlt`.
"""
ldlt(A::_SpSym;check::Bool=true,perm::Union{Nothing,AbstractVector{<:Integer}}=nothing,shift::Number=0)=LDLt(_sp_ldl(A,:none,perm,shift;check))
"""
    OxiBLAS.bunchkaufman(A::sparse, rook=false; check=true, perm=nothing) -> OxiBLAS.BunchKaufman

Multifrontal LDLᵀ factorization with 1×1 and 2×2 Bunch–Kaufman pivoting inside
the fronts and the MA57 threshold (`sparse_ldlt`). `perm` supplies a custom
fill-reducing permutation vector (`nothing` uses AMD). Only `rook=false` is
supported.
"""
bunchkaufman(A::_SpSym,rook::Bool=false;check::Bool=true,perm::Union{Nothing,AbstractVector{<:Integer}}=nothing)=
    (rook && throw(ArgumentError("rook pivoting is not available; use rook=false")); BunchKaufman(_sp_ldl(A,:bunchkaufman,perm;check)))
"""
    OxiBLAS.ldlt!(F, A; check=true, shift=0) -> F
    OxiBLAS.bunchkaufman!(F, A; check=true) -> F

Refactorizes `A` (or `A + shift*I` for `ldlt!`), which has the same pattern,
into `F`, reusing the ordering `F.p`.
"""
function ldlt!(F::LDLt{T,<:SparseLDLFactor},A::_SpSym;check::Bool=true,shift::Number=0) where T
    size(A)==size(F) || throw(DimensionMismatch("new matrix has size $(size(A)), factorization has size $(size(F))"))
    setfield!(F,:engine,_sp_ldl(A,:none,_eng(F).p,shift;check)); F
end
function bunchkaufman!(F::BunchKaufman{T,<:SparseLDLFactor},A::_SpSym;check::Bool=true) where T
    size(A)==size(F) || throw(DimensionMismatch("new matrix has size $(size(A)), factorization has size $(size(F))"))
    setfield!(F,:engine,_sp_ldl(A,:bunchkaufman,_eng(F).p;check)); F
end

# ==== QR ========================================================================
QRSparse(E::SparseQRFactor{T}) where T=QRSparse{T,typeof(E)}(E)
Base.size(F::QRSparse)=_eng(F).shape
_LA.issuccess(::QRSparse)=true
# Least squares for m ≥ n, through the engine's solve, which raises
# SingularException for a numerically rank-deficient R. B has m rows and the
# solution goes into its first n. The underdetermined case has no engine.
function _LA.ldiv!(F::QRSparse{T},B::AbstractVecOrMat) where T
    m,n=size(F); _check_rows(F,B,_ldiv_rows(F))
    m>=n || throw(ArgumentError("sparse QR solve requires m ≥ n (got $m×$n); the underdetermined solve is not available"))
    _apply_split!(X->_qr_solve!(_eng(F),view(_asmatrix(X),1:m,:)),T,B)
end
"""
    OxiBLAS.SparseQRQ{T,E} <: LinearAlgebra.AbstractQ{T}

The Q factor of a sparse QR factorization, which is a list of sparse
reflectors (`sparse_apply_q!`).
"""
struct SparseQRQ{T,E} <: _LA.AbstractQ{T}
    engine::E
end
SparseQRQ(E::SparseQRFactor{T}) where T=SparseQRQ{T,typeof(E)}(E)
Base.size(Q::SparseQRQ)=(m=_eng(Q).shape[1];(m,m))
function _spq_left!(Q::SparseQRQ,B::AbstractVecOrMat,adj::Bool)
    m=size(Q,1); size(B,1)==m || throw(DimensionMismatch("Q has size ($m, $m), operand has $(size(B,1)) rows"))
    _apply_split!(X->sparse_apply_q!(_asmatrix(X),_eng(Q);trans=adj ? :C : :N),eltype(Q),B)
end
# B·Q = (Qᴴ·Bᴴ)ᴴ: the engine applies Q on the left only.
function _spq_right!(B::AbstractMatrix,Q::SparseQRQ,adj::Bool)
    m=size(Q,1); size(B,2)==m || throw(DimensionMismatch("Q has size ($m, $m), operand has $(size(B,2)) columns"))
    X=copy(adjoint(B)); _spq_left!(Q,X,!adj); copyto!(B,adjoint(X)); B
end
_LA.lmul!(Q::SparseQRQ,B::AbstractVecOrMat)=_spq_left!(Q,B,false)
_LA.lmul!(Q::_AdjQ{<:Any,<:SparseQRQ},B::AbstractVecOrMat)=_spq_left!(parent(Q),B,true)
_LA.rmul!(B::AbstractMatrix,Q::SparseQRQ)=_spq_right!(B,Q,false)
_LA.rmul!(B::AbstractMatrix,Q::_AdjQ{<:Any,<:SparseQRQ})=_spq_right!(B,parent(Q),true)
Base.convert(::Type{_LA.AbstractQ{S}},Q::SparseQRQ{S}) where S=Q
function Base.convert(::Type{_LA.AbstractQ{S}},Q::SparseQRQ) where S
    E=_eng(Q)
    SparseQRQ(SparseQRFactor(_mul_convert(S,E.R),E.reflector_indices,[Vector{S}(v) for v in E.reflector_values],
                             Vector{real(S)}(E.tau),E.p,E.shape))
end
_LA.det(Q::SparseQRQ)=_det_tau(_eng(Q).tau)   # τ is real
function Base.getproperty(F::QRSparse,s::Symbol)
    E=_eng(F); m,n=E.shape
    s===:Q && return SparseQRQ(E)
    s===:R && return SparseMatrixCSC(E.R)[1:min(m,n),:]
    s===:prow && return collect(1:m)
    s===:pcol && return E.p
    getfield(F,s)
end
Base.propertynames(F::QRSparse,private::Bool=false)=(:R,:Q,:prow,:pcol,(private ? (:engine,) : ())...)
SparseArrays.nnz(F::QRSparse)=length(_eng(F).R.values)+sum(length,_eng(F).reflector_indices;init=0)

"""
    OxiBLAS.qr(A::sparse; ordering=:colamd) -> OxiBLAS.QRSparse

Multifrontal sparse QR factorization (`sparse_qr`, `pivot=false`), with a
fill-reducing column ordering (`:colamd`, `:amd`, `:natural` or a vector).

- Contract: `F.Q*F.R == A[F.prow,F.pcol]`, with `F.prow` the identity.
- The factorization is not rank-revealing: `F \\ b` throws
  `SingularException` on a numerically rank-deficient matrix.
"""
# The engine handles m ≥ n only (it fails with a BoundsError on a wide input).
function qr(A::_SpIn;ordering=:colamd)
    m,n=size(A)
    m>=n || throw(ArgumentError("sparse QR requires m ≥ n (got $m×$n); there is no sparse factorization of a wide matrix"))
    QRSparse(sparse_qr(_sparse_input(A);pivot=false,ordering))
end
qr(A::_SpAT;kw...)=qr(copy(A);kw...)
qr(A::_SpHerm;kw...)=qr(_sparse_full(A);kw...)

# ==== matrix-level functions ====================================================
"""
    OxiBLAS.factorize(A::sparse)

Chooses a sparse factorization from the structure of `A`, in the order of
SparseArrays' `factorize` and `\\`:

- triangular (or diagonal): `LU` with a `TriangularEngine`, the native
  sparse substitution (tested in O(nnz));
- Hermitian: `Cholesky`, or `BunchKaufman` if A is not positive definite;
- complex `Symmetric`: `BunchKaufman` (complex symmetric LDLᵀ);
- other square matrices: `LU`;
- rectangular: `QRSparse` (it solves the tall case only).
"""
function factorize(A::Union{_SpIn,_SpHerm};check::Bool=true)
    m,n=size(A)
    m==n || return qr(A)
    if A isa _SpIn
        _LA.istriu(A) && return LU(_sp_tri_engine(_LA.UpperTriangular,A,true,false))
        _LA.istril(A) && return LU(_sp_tri_engine(_LA.LowerTriangular,A,false,false))
    end
    if A isa _SpHerm || _LA.ishermitian(A)
        try
            return cholesky(A)
        catch err
            err isa PosDefException || rethrow()
        end
        return bunchkaufman(A isa _SpHerm ? A : _LA.Hermitian(A,:L);check)
    end
    lu(A;check)
end
factorize(A::_SpAT;check::Bool=true)=factorize(copy(A);check)
factorize(A::_LA.Symmetric{<:Complex,<:_SpIn};check::Bool=true)=(_LA.checksquare(A); bunchkaufman(A;check))
const _SpDetIn=Union{_SpIn,_SpAT,_SpHerm,_LA.Symmetric{<:Complex,<:_SpIn}}
det(A::_SpDetIn)=(_LA.checksquare(A); _LA.det(factorize(A;check=false)))
logdet(A::_SpDetIn)=(_LA.checksquare(A); _LA.logdet(factorize(A;check=false)))
logabsdet(A::_SpDetIn)=(_LA.checksquare(A); _LA.logabsdet(factorize(A;check=false)))
inv(A::Union{_SpIn,_SpAT,_SpHerm})=throw(ArgumentError("the inverse of a sparse matrix is usually dense; factorize it and solve with F \\ B instead"))
function isposdef(A::Union{_SpIn,_SpHerm})
    size(A,1)==size(A,2) || return false
    A isa _SpHerm || _LA.ishermitian(A) || return false
    try
        cholesky(A); return true
    catch err
        err isa PosDefException || rethrow()
        return false
    end
end

function norm(A::_SpIn,p::Real=2)
    X=_sparse_input(A); R=real(eltype(X))
    p==2 && return sparse_norm(X,:fro)
    p==Inf && return sparse_norm(X,:max)
    vals=X isa SparseMatrixCSC ? (SparseArrays.nonzeros(X),) : _mul_values(X)
    p==1 && return sum(v->sum(abs,v;init=zero(R)),vals;init=zero(R))
    p==0 && return R(sum(v->count(!iszero,v),vals;init=0))
    if p==-Inf
        nz=sum(v->count(!iszero,v),vals;init=0)
        nz<length(X) && return zero(R)
        return minimum(v->minimum(abs,v;init=R(Inf)),vals;init=R(Inf))
    end
    sum(v->sum(x->abs(x)^p,v;init=zero(R)),vals;init=zero(R))^inv(p)
end
norm(A::_SpAT,p::Real=2)=norm(parent(A),p)
norm(A::Union{_SpHerm,_LA.Symmetric{<:Complex,<:_SpIn}},p::Real=2)=norm(_sparse_hermitian(_sparse_input(parent(A)),Symbol(A.uplo)),p)
function opnorm(A::_SpIn,p::Real=2)
    (p==1 || p==Inf) || throw(ArgumentError("sparse opnorm is available for p=1 or Inf only (got p=$p)"))
    sparse_norm(_sparse_input(A),p==1 ? 1 : Inf)
end
opnorm(A::_SpAT,p::Real=2)=p==1 ? opnorm(parent(A),Inf) : p==Inf ? opnorm(parent(A),1) : opnorm(copy(A),p)
opnorm(A::Union{_SpHerm,_LA.Symmetric{<:Complex,<:_SpIn}},p::Real=2)=opnorm(_sparse_hermitian(_sparse_input(parent(A)),Symbol(A.uplo)),p)

# ==== LinearAlgebra extensions on OxiBLAS formats only ==========================
# LinearAlgebra.lu(A::CSRMatrix) and similar forward to the OxiBLAS
# functions. Matrices of the SparseArrays/Base types are never touched.
#
# `A \ B` and LinearAlgebra.ldiv! on these formats have their own methods
# (linalg_api_solve.jl), so Julia's generic AbstractMatrix `\` is no longer
# reached.
_LA.lu(A::_OxiSparse;kw...)=lu(A;kw...)
_LA.lu!(F::LU{<:Any,<:MultifrontalLUFactor},A::_SpIn;kw...)=lu!(F,A;kw...)
_LA.cholesky(A::Union{_OxiSparse,_OxiHerm};kw...)=cholesky(A;kw...)
_LA.cholesky!(F::Cholesky{<:Any,<:MultifrontalCholeskyFactor},A::Union{_SpIn,_SpHerm};kw...)=cholesky!(F,A;kw...)
_LA.ldlt(A::Union{_OxiSparse,_LA.Hermitian{<:Any,<:_OxiSparse},_LA.Symmetric{<:Any,<:_OxiSparse}};kw...)=ldlt(A;kw...)
_LA.ldlt!(F::LDLt{<:Any,<:SparseLDLFactor},A::_SpSym;kw...)=ldlt!(F,A;kw...)
_LA.bunchkaufman(A::Union{_OxiSparse,_LA.Hermitian{<:Any,<:_OxiSparse},_LA.Symmetric{<:Any,<:_OxiSparse}},rook::Bool=false;kw...)=bunchkaufman(A,rook;kw...)
_LA.qr(A::_OxiSparse;kw...)=qr(A;kw...)
_LA.factorize(A::Union{_OxiSparse,_OxiHerm})=factorize(A)
_LA.det(A::Union{_OxiSparse,_OxiHerm})=det(A)
_LA.logdet(A::Union{_OxiSparse,_OxiHerm})=logdet(A)
_LA.logabsdet(A::Union{_OxiSparse,_OxiHerm})=logabsdet(A)

# ==== SparseArrays utilities on the OxiBLAS formats =============================
"""
    SparseArrays.fkeep!(f, A::CSRMatrix / CSCMatrix)

Keeps the stored entries for which `f(i, j, x)` is true, in place: the
compressed arrays are compacted and then shrunk. The (f, A) order follows
SparseArrays 1.10+.
"""
function SparseArrays.fkeep!(f::F,A::CSRMatrix) where {F<:Function}
    rp,ci,v=A.rowptr,A.colind,A.values; k=1; lo=1
    @inbounds for i in 1:A.m
        hi=rp[i+1]-1; rp[i]=k
        for q in lo:hi
            j=ci[q]; x=v[q]
            if f(i,j,x)
                ci[k]=j; v[k]=x; k+=1
            end
        end
        lo=hi+1
    end
    rp[A.m+1]=k; resize!(ci,k-1); resize!(v,k-1); A
end
function SparseArrays.fkeep!(f::F,A::CSCMatrix) where {F<:Function}
    cp,ri,v=A.colptr,A.rowind,A.values; k=1; lo=1
    @inbounds for j in 1:A.n
        hi=cp[j+1]-1; cp[j]=k
        for q in lo:hi
            i=ri[q]; x=v[q]
            if f(i,j,x)
                ri[k]=i; v[k]=x; k+=1
            end
        end
        lo=hi+1
    end
    cp[A.n+1]=k; resize!(ri,k-1); resize!(v,k-1); A
end
"""
    SparseArrays.blockdiag(A::OxiBLAS-format, B...) -> CSCMatrix

Block-diagonal concatenation of sparse matrices in OxiBLAS formats, as a
`CSCMatrix`. It is built directly from the (row, column, value) triplets.
"""
function SparseArrays.blockdiag(A::_OxiSparse,B::_OxiSparse...)
    mats=(A,B...); T=promote_type(map(eltype,mats)...)
    rows=Int[]; cols=Int[]; vals=T[]; r0=0; c0=0
    for X in mats
        C=CSCMatrix(X)
        for j in 1:C.n, q in C.colptr[j]:C.colptr[j+1]-1
            push!(rows,r0+C.rowind[q]); push!(cols,c0+j); push!(vals,C.values[q])
        end
        r0+=C.m; c0+=C.n
    end
    CSCMatrix(COOMatrix(r0,c0,rows,cols,vals))
end

# ==== OxiBLAS.spdiagm / OxiBLAS.sparsevec =======================================
# OxiBLAS's own functions (not SparseArrays methods), so taking standard
# arguments is not piracy. They build OxiBLAS formats: CSCMatrix and the
# one-dimensional SparseTensor. Values are converted to a floating type (Int
# -> Float64), which these formats require.
function _spdiag_size(kv)
    isempty(kv) && return 0,0
    n=maximum(length(last(p))+abs(Int(first(p))) for p in kv)
    n,n
end
"""
    OxiBLAS.spdiagm(k₁ => v₁, k₂ => v₂, ...) -> CSCMatrix
    OxiBLAS.spdiagm(m, n, k₁ => v₁, ...)
    OxiBLAS.spdiagm(v) -> CSCMatrix

Sparse matrix with the vectors `vᵢ` on the diagonals `kᵢ` (k > 0 above the
main diagonal), as `SparseArrays.spdiagm`. The default size is the smallest
square that holds every diagonal; duplicate positions are summed.
"""
spdiagm(kv::Pair{<:Integer,<:AbstractVector}...)=spdiagm(_spdiag_size(kv)...,kv...)
spdiagm(v::AbstractVector)=spdiagm(0=>v)
function spdiagm(m::Integer,n::Integer,kv::Pair{<:Integer,<:AbstractVector}...)
    T=isempty(kv) ? Float64 : _la_eltype(promote_type(map(p->eltype(last(p)),kv)...))
    I=Int[]; J=Int[]; V=T[]
    for (k,v) in kv
        k=Int(k); i0,j0=k>=0 ? (0,k) : (-k,0)
        (i0+length(v)<=m && j0+length(v)<=n) || throw(DimensionMismatch("diagonal $k of length $(length(v)) does not fit in a $m×$n matrix"))
        for (t,x) in enumerate(v)
            push!(I,i0+t); push!(J,j0+t); push!(V,T(x))
        end
    end
    order=sortperm(collect(zip(J,I)))
    colptr=zeros(Int,n+1); colptr[1]=1; rowind=Int[]; vals=T[]; prev=(0,0)
    for q in order
        key=(J[q],I[q])
        if key==prev
            vals[end]+=V[q]
        else
            push!(rowind,I[q]); push!(vals,V[q]); colptr[J[q]+1]+=1; prev=key
        end
    end
    cumsum!(colptr,colptr)
    CSCMatrix(Int(m),Int(n),colptr,rowind,vals)
end
"""
    OxiBLAS.sparsevec(I, V, n=maximum(I); combine=+) -> SparseTensor{T,1}
    OxiBLAS.sparsevec(v::AbstractVector) -> SparseTensor{T,1}

OxiBLAS sparse vector, as `SparseArrays.sparsevec`:

- from indices and values, with duplicates combined by `combine`;
- from a dense vector, keeping its nonzero entries.
"""
function sparsevec(I::AbstractVector{<:Integer},V::AbstractVector,n::Integer=isempty(I) ? 0 : Int(maximum(I));combine=+)
    T=_la_eltype(eltype(V))
    SparseTensor((Int(n),),[(Int(i),) for i in I],Vector{T}(V);combine)
end
sparsevec(v::AbstractVector)=SparseTensor(Vector{_la_eltype(eltype(v))}(v))
