# SPDX-License-Identifier: Apache-2.0
# High-level API for structured and compact matrices (see the header of
# linalg_api.jl).
#
# Each structure goes to the OxiBLAS engine made for it, with no dense
# conversion:
#   - Diagonal, Bidiagonal, Tridiagonal, SymTridiagonal and BandedMatrix ->
#     band kernels (band_lu, band_cholesky, tbsv!) in O(n·bandwidth²);
#   - PackedMatrix -> packed kernels (packed_cholesky, tpsv!);
#   - triangular matrices -> substitution (trsm!/trsv!/tbsv!/tpsv!);
#   - SymTridiagonal -> eigen_tridiagonal; Bidiagonal -> the bidiagonal QR
#     kernel.
# Factorizations reuse the existing LU and Cholesky wrappers (and LDLt and
# BunchKaufman for symmetric tridiagonal and band matrices) with a compact
# engine as the parameter. Only LinearAlgebra methods on OxiBLAS's own types
# (BandedMatrix, PackedMatrix) are extended; Julia's structured types only
# get OxiBLAS methods.

const _LADiag=_LA.Diagonal
const _LABidiag=_LA.Bidiagonal
const _LATridiag=_LA.Tridiagonal
const _LASymTridiag=_LA.SymTridiagonal
const _DenseUpperTri=Union{_LA.UpperTriangular{<:Any,<:StridedMatrix},_LA.UnitUpperTriangular{<:Any,<:StridedMatrix}}
const _DenseLowerTri=Union{_LA.LowerTriangular{<:Any,<:StridedMatrix},_LA.UnitLowerTriangular{<:Any,<:StridedMatrix}}

# ---- floating copies (the factorizations never alias their input) -------------
_fvec(v::AbstractVector)=Vector{_la_eltype(eltype(v))}(v)
_float_band(A::BandedMatrix)=BandedMatrix(Matrix{_la_eltype(eltype(A))}(A.data),A.m,A.kl,A.ku,A.kind,A.upper,A.unit)
_float_packed(A::PackedMatrix)=PackedMatrix(Vector{_la_eltype(eltype(A))}(A.data),A.n,A.upper,A.kind,A.unit)
_kd(A::BandedMatrix)=max(A.kl,A.ku)

# General band storage of a square band matrix given by getindex, which
# reflects the symmetric/Hermitian kinds. Costs O(n·(kl+ku)).
function _band_general(A::AbstractMatrix,kl::Int,ku::Int)
    n=_square(A); T=_la_eltype(eltype(A)); data=zeros(T,kl+ku+1,n)
    @inbounds for j in 1:n, i in max(1,j-ku):min(n,j+kl)
        data[ku+1+i-j,j]=A[i,j]
    end
    BandedMatrix(data,n;kl,ku,kind=:general)
end
_band_lu_input(A::BandedMatrix)=A.kind===:general ? _float_band(A) :
    _band_general(A,A.kind===:triangular ? A.kl : _kd(A),A.kind===:triangular ? A.ku : _kd(A))

# Triangular band storage of a bidiagonal matrix (2×n).
function _bidiag_band(B::_LABidiag)
    n=size(B,1); T=_la_eltype(eltype(B)); upper=B.uplo=='U'; data=zeros(T,2,n)
    for j in 1:n
        data[upper ? 2 : 1,j]=B.dv[j]
        if upper
            j>1 && (data[1,j]=B.ev[j-1])
        else
            j<n && (data[2,j]=B.ev[j])
        end
    end
    BandedMatrix(data,n;kl=upper ? 0 : 1,ku=upper ? 1 : 0,kind=:triangular,uplo=upper ? :U : :L)
end
# Lower triangle of a symmetric tridiagonal or symmetric/Hermitian band
# matrix as an OxiBLAS CSC matrix, for the multifrontal LDLᵀ. Costs O(n·kd).
function _band_lower_csc(A::AbstractMatrix,kd::Int)
    n=_square(A); T=_la_eltype(eltype(A)); rows=Int[]; cols=Int[]; vals=T[]
    for j in 1:n, i in j:min(n,j+kd)
        v=A[i,j]; (i==j || !iszero(v)) || continue
        push!(rows,i); push!(cols,j); push!(vals,v)
    end
    CSCMatrix(COOMatrix(n,n,rows,cols,vals))
end

# ==== substitution engine (triangular, bidiagonal, diagonal) ====================
"""
    OxiBLAS.TriangularEngine{T,S,K}

Engine of an `OxiBLAS.LU` for a matrix that is already triangular: dense
(`UpperTriangular`…), bidiagonal, diagonal, triangular `BandedMatrix` or
triangular `PackedMatrix`.

- Solves are substitutions with the matching kernel: `trsm!`/`trsv!`
  (dense), `tbsv!` (band; a bidiagonal is stored as a 2×n band), `tpsv!`
  (packed), or a division (diagonal). They cost O(n²) for dense storage and
  O(n·k) for a band.
- For an upper (or diagonal) matrix, it is the partial-pivoting LU
  (L = I, p = identity).
- For a lower matrix, it is the LU without pivoting, L = A·D⁻¹ and U = D. It
  is used only by `factorize`, since `lu` (RowMaximum) has to pivot.
- `src` is the engine's own copy of the input, and `kernel` the storage
  handed to the kernel.
"""
struct TriangularEngine{T,S,K}
    src::S
    kernel::K
    upper::Bool
    unit::Bool
    info::Int
end
function _first_zero_pivot(A,unit::Bool)
    unit && return 0
    for i in 1:size(A,1)
        iszero(A[i,i]) && return i
    end
    0
end
function TriangularEngine(src::AbstractMatrix,kernel,upper::Bool,unit::Bool)
    T=eltype(src)
    TriangularEngine{T,typeof(src),typeof(kernel)}(src,kernel,upper,unit,_first_zero_pivot(src,unit))
end
_tri_wrap(::_LA.UpperTriangular,M)=_LA.UpperTriangular(M)
_tri_wrap(::_LA.UnitUpperTriangular,M)=_LA.UnitUpperTriangular(M)
_tri_wrap(::_LA.LowerTriangular,M)=_LA.LowerTriangular(M)
_tri_wrap(::_LA.UnitLowerTriangular,M)=_LA.UnitLowerTriangular(M)
function _tri_engine(A::Union{_DenseUpperTri,_DenseLowerTri})
    M=Matrix{_la_eltype(eltype(A))}(parent(A))
    TriangularEngine(_tri_wrap(A,M),M,A isa _DenseUpperTri,A isa Union{_LA.UnitUpperTriangular,_LA.UnitLowerTriangular})
end
_tri_engine(D::_LADiag)=(d=_fvec(D.diag); TriangularEngine(_LA.Diagonal(d),d,true,false))
function _tri_engine(B::_LABidiag)
    C=_LA.Bidiagonal(_fvec(B.dv),_fvec(B.ev),B.uplo=='U' ? :U : :L)
    TriangularEngine(C,_bidiag_band(C),B.uplo=='U',false)
end
_tri_engine(A::BandedMatrix)=(B=_float_band(_pkcheck(A,:triangular)); TriangularEngine(B,B,B.upper,B.unit))
_tri_engine(A::PackedMatrix)=(P=_float_packed(_pkcheck(A,:triangular)); TriangularEngine(P,P,P.upper,P.unit))

_tri_kernel!(d::AbstractVector,X::AbstractVecOrMat,E,tr)=
    (for j in axes(X,2), i in axes(X,1); X[i,j]/=(tr===:C ? conj(d[i]) : d[i]); end; X)
_tri_kernel!(M::StridedMatrix,X::AbstractVecOrMat,E,tr)=_trisolve!(X,M,E.upper ? :U : :L,tr,E.unit)
_tri_kernel!(K::BandedMatrix,X::AbstractVecOrMat,E,tr)=(for j in axes(X,2); tbsv!(view(X,:,j),K;trans=tr); end; X)
_tri_kernel!(K::PackedMatrix,X::AbstractVecOrMat,E,tr)=(for j in axes(X,2); tpsv!(view(X,:,j),K;trans=tr); end; X)

LU(E::TriangularEngine{T}) where T=LU{T,typeof(E)}(E)
Base.size(F::LU{<:Any,<:TriangularEngine})=size(_eng(F).src)
_LA.issuccess(F::LU{<:Any,<:TriangularEngine})=_eng(F).info==0
for (W,tr) in ((:LU,:(:N)),(:_AdjF,:(:C)),(:_TrF,:(:T)))
    S=W===:LU ? :(LU{T,<:TriangularEngine}) : :($W{<:Any,<:LU{T,<:TriangularEngine}})
    @eval function _LA.ldiv!(Fw::$S,B::AbstractVecOrMat) where T
        E=_eng($(W===:LU ? :Fw : :(parent(Fw)))); _check_rows(Fw,B)
        E.info==0 || throw(SingularException(E.info))
        _apply_split!(X->_tri_kernel!(E.kernel,X,E,$tr),T,B)
    end
end
function _LA.logabsdet(F::LU{T,<:TriangularEngine}) where T
    E=_eng(F); d=zero(real(T)); s=one(T)
    E.unit && return (d,s)
    for i in 1:size(E.src,1)
        u=E.src[i,i]; d+=log(abs(u)); s*=sign(u)
    end
    (d,s)
end
# Unit lower factor A·D⁻¹ of a lower matrix, in the input's own structure.
_unit_lower(A::_LA.LowerTriangular,d)=(M=copy(parent(A)); n=size(M,1);
    for j in 1:n, i in j+1:n; M[i,j]/=d[j]; end; _LA.UnitLowerTriangular(M))
_unit_lower(B::_LABidiag,d)=_LA.Bidiagonal(ones(eltype(d),length(d)),B.ev./view(d,1:length(d)-1),:L)
function _unit_lower(A::BandedMatrix,d)
    D=copy(A.data)
    for j in axes(D,2), r in 2:min(A.kl+1,size(D,1)); D[r,j]/=d[j]; end
    BandedMatrix(D,A.m,A.kl,0,:triangular,false,true)
end
function _unit_lower(A::PackedMatrix,d)
    P=PackedMatrix(copy(A.data),A.n,false,:triangular,true)
    for j in 1:A.n, i in j+1:A.n; P[i,j]=A[i,j]/d[j]; end
    P
end
function Base.getproperty(F::LU{T,<:TriangularEngine},s::Symbol) where T
    E=_eng(F); n=size(E.src,1)
    if s===:L || s===:U
        E.upper && return s===:L ? _LA.Diagonal(ones(T,n)) : E.src
        E.unit && return s===:L ? E.src : _LA.Diagonal(ones(T,n))
        E.info==0 || throw(SingularException(E.info))
        d=T[E.src[i,i] for i in 1:n]
        return s===:L ? _unit_lower(E.src,d) : _LA.Diagonal(d)
    end
    s===:p && return collect(1:n)
    getfield(F,s)
end
Base.propertynames(F::LU{<:Any,<:TriangularEngine},private::Bool=false)=(:L,:U,:p,(private ? (:engine,) : ())...)

# ==== band LU (Diagonal, Bidiagonal, Tridiagonal, SymTridiagonal, BandedMatrix) =
LU(E::BandLUFactor{T}) where T=LU{T,typeof(E)}(E)
Base.size(F::LU{<:Any,<:BandLUFactor})=(_eng(F).n,_eng(F).n)
_LA.issuccess(F::LU{<:Any,<:BandLUFactor})=true   # band_lu throws SingularException
for (W,tr) in ((:LU,:(:N)),(:_AdjF,:(:C)),(:_TrF,:(:T)))
    S=W===:LU ? :(LU{T,<:BandLUFactor}) : :($W{<:Any,<:LU{T,<:BandLUFactor}})
    @eval function _LA.ldiv!(Fw::$S,B::AbstractVecOrMat) where T
        E=_eng($(W===:LU ? :Fw : :(parent(Fw)))); _check_rows(Fw,B)
        _apply_split!(X->copyto!(X,solve(E,X;trans=$tr)),T,B)
    end
end
function _LA.logabsdet(F::LU{T,<:BandLUFactor}) where T
    E=_eng(F); d=zero(real(T)); s=one(T)
    for j in 1:E.n
        u=_bandlu_get(E,j,j); d+=log(abs(u)); s*=sign(u)
        E.pivots[j]!=j && (s=-s)
    end
    (d,s)
end
# Explicit P·A = L·U from the GBTRF-style storage, where the multipliers of
# step j are kept in elimination order. Entry (i,j) of L lands on row σⱼ(i),
# where σⱼ composes the later row swaps; U rows are final once eliminated.
# O(n·(kl+ku)), returned as sparse matrices.
function _band_lu_factors(E::BandLUFactor{T}) where T
    n,kl,ku=E.n,E.kl,E.ku; σ=collect(1:n)
    Li=Int[]; Lj=Int[]; Lv=T[]
    for j in n:-1:1
        for i in j+1:min(n,j+kl)
            v=_bandlu_get(E,i,j); iszero(v) && continue
            push!(Li,σ[i]); push!(Lj,j); push!(Lv,v)
        end
        p=E.pivots[j]; σ[j],σ[p]=σ[p],σ[j]
    end
    append!(Li,1:n); append!(Lj,1:n); append!(Lv,ones(T,n))
    Ui=Int[]; Uj=Int[]; Uv=T[]
    for j in 1:n, i in max(1,j-kl-ku):j
        v=_bandlu_get(E,i,j); (i==j || !iszero(v)) || continue
        push!(Ui,i); push!(Uj,j); push!(Uv,v)
    end
    SparseArrays.sparse(Li,Lj,Lv,n,n),SparseArrays.sparse(Ui,Uj,Uv,n,n),invperm(σ)
end
function Base.getproperty(F::LU{T,<:BandLUFactor},s::Symbol) where T
    (s===:L || s===:U || s===:p) || return getfield(F,s)
    L,U,p=_band_lu_factors(_eng(F))
    s===:L ? L : s===:U ? U : p
end
Base.propertynames(F::LU{<:Any,<:BandLUFactor},private::Bool=false)=(:L,:U,:p,(private ? (:engine,) : ())...)

"""
    OxiBLAS.lu(A::Tridiagonal / SymTridiagonal / Bidiagonal / Diagonal / BandedMatrix)

Band LU with partial pivoting (`band_lu`). It works in (2kl+ku+1)×n compact
storage, which includes the fill from pivoting, in O(n·kl·(kl+ku)) time
(O(n) for a tridiagonal).

- `F.L`, `F.U` (sparse) and `F.p` are rebuilt on demand so that
  `F.L*F.U == A[F.p,:]`.
- A singular matrix throws `SingularException` whatever the value of
  `check`.
- There is no `lu!`: band storage cannot hold the fill.
"""
lu(A::_LATridiag;check::Bool=true)=LU(band_lu(_band_general(A,1,1)))
lu(A::_LASymTridiag;check::Bool=true)=LU(band_lu(_band_general(A,1,1)))
lu(A::_LABidiag;check::Bool=true)=LU(band_lu(_band_general(A,A.uplo=='U' ? 0 : 1,A.uplo=='U' ? 1 : 0)))
lu(A::_LADiag;check::Bool=true)=LU(band_lu(_band_general(A,0,0)))
lu(A::BandedMatrix;check::Bool=true)=LU(band_lu(_band_lu_input(A)))
"""
    OxiBLAS.lu(A::UpperTriangular / UnitUpperTriangular / packed upper triangular)

The partial-pivoting LU of an upper triangular matrix is `L = I`, `U = A`. It
is returned without elimination, on a copy, with substitution solves
(`TriangularEngine`). A lower triangular matrix may need row swaps, so a dense
one goes through the dense LU.
"""
lu(A::_DenseUpperTri,::_LA.RowMaximum=_LA.RowMaximum();check::Bool=true)=_tri_lu(_tri_engine(A),check)
function lu(A::PackedMatrix;check::Bool=true)
    (A.kind===:triangular && A.upper) || throw(ArgumentError("packed LU needs an upper triangular PackedMatrix; there is no general packed LU engine"))
    _tri_lu(_tri_engine(A),check)
end
function _tri_lu(E::TriangularEngine,check::Bool)
    check && E.info!=0 && throw(SingularException(E.info))
    LU(E)
end

# ==== Cholesky: band, packed, diagonal ==========================================
Cholesky(E::Union{BandCholeskyFactor{T},PackedCholeskyFactor{T}},info::Integer=0) where T=Cholesky{T,typeof(E)}(E,Int(info))
Cholesky(E::_LADiag{T},info::Integer=0) where T=Cholesky{T,typeof(E)}(E,Int(info))
Base.size(F::Cholesky{<:Any,<:BandCholeskyFactor})=(_eng(F).n,_eng(F).n)
Base.size(F::Cholesky{<:Any,<:PackedCholeskyFactor})=size(_eng(F).L)
Base.size(F::Cholesky{<:Any,<:_LADiag})=size(_eng(F))
function _LA.ldiv!(F::Cholesky{T,<:Union{BandCholeskyFactor,PackedCholeskyFactor}},B::AbstractVecOrMat) where T
    E=_chol_ready(F); _check_rows(F,B)
    _apply_split!(X->copyto!(X,solve(E,X)),T,B)
end
function _LA.ldiv!(F::Cholesky{T,<:_LADiag},B::AbstractVecOrMat) where T
    l=_chol_ready(F).diag; _check_rows(F,B)
    _apply_split!(T,B) do X
        for j in axes(X,2), i in axes(X,1); X[i,j]/=abs2(l[i]); end
    end
end
_chol_diag(E::BandCholeskyFactor,i)=E.data[1,i]
_chol_diag(E::PackedCholeskyFactor,i)=E.L[i,i]
_chol_diag(E::_LADiag,i)=E.diag[i]
function _LA.logabsdet(F::Cholesky{T,<:Union{BandCholeskyFactor,PackedCholeskyFactor,_LADiag}}) where T
    E=_chol_ready(F); d=zero(real(T))
    for i in 1:size(F,1); d+=log(real(_chol_diag(E,i))); end
    (2d,one(T))
end
# L in the input's compact structure, shared with the engine (no copy).
_chol_L(E::BandCholeskyFactor)=BandedMatrix(E.data,E.n,E.kd,0,:triangular,false,false)
_chol_L(E::PackedCholeskyFactor)=E.L
_chol_L(E::_LADiag)=E
function Base.getproperty(F::Cholesky{<:Any,<:Union{BandCholeskyFactor,PackedCholeskyFactor,_LADiag}},s::Symbol)
    (s===:L || s===:UL) && return _chol_L(_eng(F))
    s===:U && return adjoint(_chol_L(_eng(F)))
    s===:uplo && return 'L'
    getfield(F,s)
end
Base.propertynames(F::Cholesky{<:Any,<:Union{BandCholeskyFactor,PackedCholeskyFactor,_LADiag}},private::Bool=false)=
    (:L,:U,:UL,(private ? (:uplo,:info,:engine) : ())...)

# The engines throw PosDefException; check=false keeps a zero engine of the
# right shape and the failing index, as the dense path does.
function _chol_compact(f,blank,check::Bool)
    try
        return Cholesky(f())
    catch err
        (check || !(err isa PosDefException)) && rethrow()
        return Cholesky(blank(),err.info)
    end
end
function _sym_band(A::_LASymTridiag)
    n=size(A,1); T=_la_eltype(eltype(A)); data=zeros(T,2,n)
    for j in 1:n
        data[1,j]=A.dv[j]; j<n && (data[2,j]=A.ev[j])
    end
    BandedMatrix(data,n;kl=1,kind=:symmetric,uplo=:L)
end
"""
    OxiBLAS.cholesky(A::SymTridiagonal{<:Real} / BandedMatrix / PackedMatrix / Diagonal; check=true)

Cholesky factorization in the storage of `A`:

- band (`band_cholesky`, O(n·kd²)): `F.L` is a lower triangular `BandedMatrix`;
- packed (`packed_cholesky`): `F.L` is a triangular `PackedMatrix`;
- diagonal: `F.L` is a `Diagonal`.

`A` must be symmetric (real) or Hermitian in its own storage; a general or
triangular `BandedMatrix`/`PackedMatrix` gets an `ArgumentError`.
"""
cholesky(A::_LASymTridiag{<:Real};check::Bool=true)=(B=_sym_band(A); n=size(B,1);
    _chol_compact(()->band_cholesky(B),()->BandCholeskyFactor(zeros(eltype(B),2,n),n,1),check))
function cholesky(A::BandedMatrix;check::Bool=true)
    B=_float_band(A); n=size(B,1); kd=_kd(B)
    _chol_compact(()->band_cholesky(B),()->BandCholeskyFactor(zeros(eltype(B),kd+1,n),n,kd),check)
end
function cholesky(A::PackedMatrix;check::Bool=true)
    P=_float_packed(A); n=size(P,1)
    _chol_compact(()->packed_cholesky(P),
                  ()->PackedCholeskyFactor(PackedMatrix(zeros(eltype(P),length(P.data)),n;uplo=:L,kind=:triangular)),check)
end
function cholesky(A::_LADiag;check::Bool=true)
    d=_fvec(A.diag); T=eltype(d); l=similar(d)
    for i in eachindex(d)
        x=d[i]
        if !(isreal(x) && real(x)>0 && isfinite(real(x)))
            check && throw(PosDefException(i))
            return Cholesky(_LA.Diagonal(zeros(T,length(d))),i)
        end
        l[i]=sqrt(real(x))
    end
    Cholesky(_LA.Diagonal(l))
end

# ==== LDLᵀ / Bunch–Kaufman: symmetric tridiagonal and band ======================
# Multifrontal LDLᵀ in natural order on the lower triangle: the elimination
# tree is a chain, the fill stays in the band, and symmetry is kept. Complex
# symmetric (not Hermitian) input gets the LDLᵀ form.
_band_ldl(A::_LASymTridiag,pivot)=sparse_ldlt(_band_lower_csc(A,1);uplo=:L,hermitian=eltype(A)<:Real,pivot,ordering=:natural)
function _band_ldl(A::BandedMatrix,pivot)
    A.kind in (:symmetric,:hermitian) || throw(ArgumentError("LDLᵀ needs symmetric or Hermitian band storage"))
    sparse_ldlt(_band_lower_csc(A,_kd(A));uplo=:L,hermitian=A.kind===:hermitian || eltype(A)<:Real,pivot,ordering=:natural)
end
"""
    OxiBLAS.ldlt(A::SymTridiagonal / symmetric BandedMatrix)
    OxiBLAS.bunchkaufman(A::SymTridiagonal / symmetric BandedMatrix)

LDLᵀ without pivoting (as LinearAlgebra's `ldlt(::SymTridiagonal)`), or with
Bunch–Kaufman pivoting. Both use the multifrontal engine (`sparse_ldlt`) in
natural order, in O(n·kd²) time and memory.
"""
ldlt(A::Union{_LASymTridiag,BandedMatrix})=LDLt(_band_ldl(A,:none))
bunchkaufman(A::Union{_LASymTridiag,BandedMatrix},rook::Bool=false;check::Bool=true)=
    (rook && throw(ArgumentError("rook pivoting is not available; use rook=false")); BunchKaufman(_band_ldl(A,:bunchkaufman)))

# ==== factorize =================================================================
"""
    OxiBLAS.factorize(A) for structured matrices

| Input | Result |
|---|---|
| `Diagonal`, `Bidiagonal`, triangular (dense, band, packed) | `LU` with `TriangularEngine`: substitution, no elimination |
| `Tridiagonal`, general `BandedMatrix` | band `LU` |
| `SymTridiagonal`, symmetric/Hermitian `BandedMatrix` | band `Cholesky` if A is positive definite (real or Hermitian), otherwise `BunchKaufman` (multifrontal, natural order) |
| Hermitian/real symmetric `PackedMatrix` | packed `Cholesky` |

An indefinite `PackedMatrix` gets an `ArgumentError`: no packed indefinite
engine exists.
"""
# Separate methods: a Union mixing dense triangulars with other types would be
# ambiguous with factorize(::_DenseIn).
factorize(A::Union{_LADiag,_LABidiag})=LU(_tri_engine(A))
factorize(A::Union{_DenseUpperTri,_DenseLowerTri})=LU(_tri_engine(A))
factorize(A::_LATridiag)=lu(A)
function _try_chol(f)
    try
        return f()
    catch err
        err isa Union{PosDefException,ArgumentError} || rethrow()
        return nothing
    end
end
function factorize(A::_LASymTridiag)
    if eltype(A)<:Real
        C=_try_chol(()->cholesky(A)); C===nothing || return C
    end
    bunchkaufman(A)
end
function factorize(A::BandedMatrix)
    A.kind===:general && return lu(A)
    A.kind===:triangular && return LU(_tri_engine(A))
    C=_try_chol(()->cholesky(A)); C===nothing || return C
    bunchkaufman(A)
end
function factorize(A::PackedMatrix)
    A.kind===:triangular && return LU(_tri_engine(A))
    C=cholesky(A;check=false)
    _LA.issuccess(C) && return C
    throw(ArgumentError("PackedMatrix is not positive definite and there is no packed indefinite (Bunch–Kaufman) engine"))
end

# ==== spectra ===================================================================
# _eigsortby: linalg_api.jl.
"""
    OxiBLAS.eigen(A::SymTridiagonal{<:Real}) / eigvals

Implicit-QR symmetric tridiagonal eigensolver (`eigen_tridiagonal`), with
ascending values and O(n) memory without vectors.
"""
eigen(A::_LASymTridiag{<:Real})=(n=size(A,1); E=eigen_tridiagonal(_fvec(A.dv),_fvec(view(A.ev,1:max(n-1,0))));
    Eigen(E.values,E.vectors))
eigvals(A::_LASymTridiag{<:Real})=(n=size(A,1); eigen_tridiagonal(_fvec(A.dv),_fvec(view(A.ev,1:max(n-1,0)));vectors=false).values)
"""
    OxiBLAS.eigen(D::Diagonal; sortby) / eigvals

The values are the diagonal, sorted as in LinearAlgebra (real part, then
imaginary part); the vectors are the matching permutation of the identity.
With `sortby=nothing`, the values stay unsorted and the vectors are a
`Diagonal` (O(n)).
"""
function eigen(D::_LADiag;sortby::Union{Function,Nothing}=_eigsortby)
    λ=_fvec(D.diag); n=length(λ); T=eltype(λ)
    sortby===nothing && return Eigen(λ,_LA.Diagonal(ones(T,n)))
    p=sortperm(λ;by=sortby); V=zeros(T,n,n)
    for k in 1:n; V[p[k],k]=one(T); end
    Eigen(λ[p],V)
end
eigvals(D::_LADiag;sortby::Union{Function,Nothing}=_eigsortby)=(λ=_fvec(D.diag); sortby===nothing ? λ : sort!(λ;by=sortby))
# The eigenvalues of a triangular matrix are its diagonal, as in LinearAlgebra.
_diagvals(A)=_la_eltype(eltype(A))[A[i,i] for i in 1:size(A,1)]
eigvals(A::Union{_DenseUpperTri,_DenseLowerTri})=_diagvals(A)
eigvals(A::_LABidiag)=_diagvals(A)
"""
    OxiBLAS.eigen(A::UpperTriangular / UnitUpperTriangular)

The values are the diagonal; the vectors come from back substitution on the
triangular matrix (`trevc_right`), with no QR iteration. Not available for a
lower triangular matrix.
"""
function eigen(A::_DenseUpperTri)
    M=Matrix{_la_eltype(eltype(A))}(A)   # zero strict lower triangle, as trevc_right reads
    Eigen([M[i,i] for i in 1:size(M,1)],trevc_right(M))   # real vectors for real input: no 2×2 blocks
end

# Bidiagonal SVD: B = P·|B|·Qᴴ with unit diagonal P and Q, so the
# bidiagonal QR kernel runs on the moduli |d|, |e| (O(n) memory without
# vectors). A lower B is the transpose of the upper one with the same d, e.
function _bidiag_svd(B::_LABidiag,vectors::Bool)
    T=_la_eltype(eltype(B)); R=real(T); n=size(B,1)
    d=R[abs(x) for x in B.dv]; e=R[abs(x) for x in B.ev]
    U0=vectors ? _identity(R,n) : nothing; V0=vectors ? _identity(R,n) : nothing
    _bidiagonal_qr!(d,e,U0,V0,_floating_iterations(R,160)*max(1,n)^2,8eps(R))
    order=sortperm(d;rev=true); S=d[order]
    vectors || return S,nothing,nothing
    ph(x)=iszero(x) ? one(T) : T(x/abs(x))
    p=Vector{T}(undef,n); q=Vector{T}(undef,n)
    n>0 && (q[1]=one(T); p[1]=ph(B.dv[1]))
    for j in 1:n-1
        q[j+1]=conj(ph(B.ev[j]))*p[j]; p[j+1]=ph(B.dv[j+1])*q[j+1]
    end
    U=T.(U0[:,order]); V=T.(V0[:,order])
    for j in 1:n, i in 1:n; U[i,j]*=p[i]; V[i,j]*=q[i]; end
    B.uplo=='U' ? (S,U,V) : (S,conj.(V),conj.(U))
end
"""
    OxiBLAS.svd(B::Bidiagonal) / svdvals(B::Bidiagonal)
    OxiBLAS.svd(D::Diagonal) / svdvals(D::Diagonal)

Bidiagonal: Golub–Kahan QR (the `svd_bidiagonal` kernel) applied directly
to (|d|,|e|) after removing the phases, with no Householder reduction and
no dense copy. Diagonal: sorted moduli, with vectors made of the phases and
the permutation, as in LinearAlgebra.
"""
function svd(B::_LABidiag)
    S,U,V=_bidiag_svd(B,true); SVD(SVDFactor(U,S,Matrix(adjoint(V)),0))
end
svdvals(B::_LABidiag)=_bidiag_svd(B,false)[1]
function svd(D::_LADiag)
    d=_fvec(D.diag); T=eltype(d); n=length(d); s=abs.(d); p=sortperm(s;rev=true)
    U=zeros(T,n,n); V=zeros(T,n,n)
    for k in 1:n
        j=p[k]; V[j,k]=one(T); U[j,k]=iszero(d[j]) ? one(T) : d[j]/s[j]
    end
    SVD(SVDFactor(U,s[p],Matrix(adjoint(V)),0))
end
svdvals(D::_LADiag)=sort!(abs.(_fvec(D.diag));rev=true)

# ==== functions on structured matrices ==========================================
# Dense triangulars get their own methods (a joint Union would be ambiguous
# with det(::_DenseIn)).
const _StructIn=Union{_LADiag,_LABidiag,_LATridiag,_LASymTridiag,BandedMatrix,PackedMatrix}
for f in (:det,:logdet,:logabsdet)
    @eval $f(A::_StructIn)=_LA.$f(factorize(A))
    @eval $f(A::Union{_DenseUpperTri,_DenseLowerTri})=_LA.$f(factorize(A))
end
"""
    OxiBLAS.inv(A) for structured matrices

- `Diagonal` → `Diagonal` (O(n)).
- Triangular → the same triangular wrapper, from the substitution solves of
  the identity.
- Any other structure → the dense inverse from its structured factorization
  (the inverse of a band matrix is dense).
"""
function inv(D::_LADiag)
    d=_fvec(D.diag)
    for i in eachindex(d); iszero(d[i]) && throw(SingularException(i)); end
    _LA.Diagonal(Base.inv.(d))
end
inv(A::Union{_DenseUpperTri,_DenseLowerTri})=(X=Base.inv(factorize(A));
    A isa _DenseUpperTri ? _LA.UpperTriangular(X) : _LA.LowerTriangular(X))
inv(A::Union{_LABidiag,_LATridiag,_LASymTridiag,BandedMatrix,PackedMatrix})=Base.inv(factorize(A))
isposdef(A::Union{_LADiag,_LASymTridiag{<:Real},BandedMatrix,PackedMatrix})=
    (A isa Union{BandedMatrix,PackedMatrix} && !(A.kind in (:symmetric,:hermitian))) ? false :
    (A isa Union{BandedMatrix,PackedMatrix} && A.kind===:symmetric && !(eltype(A)<:Real)) ? false :
    _LA.issuccess(cholesky(A;check=false))
"""
    OxiBLAS.rank(A), OxiBLAS.cond(A) for `Diagonal` and `Bidiagonal`

Computed from the structured singular values (`svdvals`): O(n log n) for
`Diagonal`, O(n) memory for `Bidiagonal`.
"""
function rank(A::Union{_LADiag,_LABidiag};atol::Real=0,rtol::Real=_la_rtol(A,atol))
    s=svdvals(A); isempty(s) && return 0
    count(>(max(atol,rtol*s[1])),s)
end
function cond(A::Union{_LADiag,_LABidiag},p::Real=2)
    p==2 || throw(ArgumentError("structured cond is available for p=2 only"))
    s=svdvals(A); isempty(s) && return zero(eltype(s))
    iszero(s[end]) ? oftype(s[1],Inf) : s[1]/s[end]
end

# ==== Hermitian/Symmetric wrappers of structured matrices =======================
# Hermitian(Tridiagonal), Symmetric(Bidiagonal), and so on: the triangle the
# wrapper selects becomes a symmetric or Hermitian band matrix (kd = 1, or 0
# for a Diagonal) in O(n). All the band methods above apply to it.
const _BandStruct=Union{_LADiag,_LABidiag,_LATridiag,_LASymTridiag}
const _StructHS=Union{_LA.Hermitian{<:Any,<:_BandStruct},_LA.Symmetric{<:Any,<:_BandStruct}}
function _struct_band(H::_StructHS)
    n=size(H,1); kd=parent(H) isa _LADiag ? 0 : 1; T=_la_eltype(eltype(H)); data=zeros(T,kd+1,n)
    for j in 1:n
        data[1,j]=H[j,j]   # getindex of the wrapper reflects the selected triangle
        kd==1 && j<n && (data[2,j]=H[j+1,j])
    end
    kind=H isa _LA.Hermitian || T<:Real ? :hermitian : :symmetric
    BandedMatrix(data,n;kl=kd,kind,uplo=:L)
end
"""
    OxiBLAS.cholesky / ldlt / bunchkaufman / lu / factorize / det / isposdef
        on Hermitian(S) or Symmetric(S), with S a Diagonal, Bidiagonal,
        Tridiagonal or SymTridiagonal

Go through the triangle that the wrapper selects, converted in O(n) to a
symmetric or Hermitian `BandedMatrix`, so to the band engines
(`band_cholesky`, multifrontal LDLᵀ, `band_lu`). For a real symmetric
input, `eigen`/`eigvals` go to `eigen_tridiagonal`.
"""
cholesky(H::_StructHS;check::Bool=true)=cholesky(_struct_band(H);check)
ldlt(H::_StructHS)=ldlt(_struct_band(H))
bunchkaufman(H::_StructHS,rook::Bool=false;check::Bool=true)=bunchkaufman(_struct_band(H),rook;check)
lu(H::_StructHS;check::Bool=true)=lu(_struct_band(H);check)
factorize(H::_StructHS)=factorize(_struct_band(H))
for f in (:det,:logdet,:logabsdet)
    @eval $f(H::_StructHS)=$f(_struct_band(H))
end
inv(H::_StructHS)=inv(_struct_band(H))
isposdef(H::_StructHS)=isposdef(_struct_band(H))
const _RealStructHS=Union{_LA.Hermitian{<:Real,<:_BandStruct},_LA.Symmetric{<:Real,<:_BandStruct}}
function _struct_symtridiag(H::_RealStructHS)
    n=size(H,1); T=_la_eltype(eltype(H))
    _LA.SymTridiagonal(T[H[i,i] for i in 1:n],T[H[i+1,i] for i in 1:n-1])
end
eigen(H::_RealStructHS)=eigen(_struct_symtridiag(H))
eigvals(H::_RealStructHS)=eigvals(_struct_symtridiag(H))

# ==== LinearAlgebra extensions: OxiBLAS compact types only ======================
# As for the sparse formats, Julia's generic `A \ b` on a BandedMatrix or a
# PackedMatrix (istril/istriu, then lu) now reaches these engines.
_LA.lu(A::BandedMatrix;kw...)=lu(A;kw...)
# Not LinearAlgebra.lu(::PackedMatrix): only the upper triangular packed
# matrix has an LU, and extending it would break the generic `\` on the
# other packed kinds.
_LA.cholesky(A::Union{BandedMatrix,PackedMatrix};kw...)=cholesky(A;kw...)
_LA.ldlt(A::BandedMatrix)=ldlt(A)
_LA.bunchkaufman(A::BandedMatrix,rook::Bool=false;kw...)=bunchkaufman(A,rook;kw...)
_LA.factorize(A::Union{BandedMatrix,PackedMatrix})=factorize(A)
_LA.det(A::Union{BandedMatrix,PackedMatrix})=det(A)
_LA.logdet(A::Union{BandedMatrix,PackedMatrix})=logdet(A)
_LA.logabsdet(A::Union{BandedMatrix,PackedMatrix})=logabsdet(A)
_LA.isposdef(A::Union{BandedMatrix,PackedMatrix})=isposdef(A)

# ==== matrix functions exp/log/sqrt/sin/cos =====================================
"""
    OxiBLAS.exp(A), OxiBLAS.log(A), OxiBLAS.sqrt(A), OxiBLAS.sin(A), OxiBLAS.cos(A)

Matrix functions on the native engines:

- `matrix_exp`: Padé scaling and squaring;
- `matrix_log`: complex Schur, then inverse scaling and squaring;
- `matrix_sqrt`: complex Schur, then triangular recurrence;
- `matrix_sin`, `matrix_cos`: through exp(±iA).

On a number they are `Base`'s functions. No other type falls back to
Base/LinearAlgebra.

The engines exploit no structure: they make one working copy of any
`AbstractMatrix`, which is passed to them as is. The result keeps the
structure that f preserves:

| Input | Result |
|---|---|
| `Diagonal` | `Diagonal`, entry by entry (O(n)) |
| upper/lower triangular (dense, `Bidiagonal`, triangular band/packed) | `UpperTriangular`/`LowerTriangular` |
| Hermitian (wrapper, Hermitian band/packed) | exp/sin/cos: `Hermitian`. log/sqrt: Hermitian eigendecomposition V·f(λ)·Vᴴ, as in LinearAlgebra; `Hermitian` if A is positive definite (log) or semidefinite (sqrt, every λ ≥ -n·eps·max abs(λ)), otherwise a full complex matrix with the branch f(complex(λ)), never forced Hermitian |
| symmetric (`Symmetric`, `SymTridiagonal`, symmetric band/packed) | `Symmetric` (f(A)ᵀ = f(Aᵀ)) |
| `Tridiagonal`, general `BandedMatrix`, other | full matrix (dense in general) |

The imaginary part is dropped only when the result is real
mathematically:
- sin and cos of a real matrix;
- log of a real symmetric positive definite matrix, sqrt of a real
  symmetric positive semidefinite one;
- log/sqrt of a real triangular matrix with a positive diagonal.
"""
exp

# Structure of the result: :upper, :lower, :hermitian, :symmetric or :general.
_mf_kind(::_DenseUpperTri)=:upper
_mf_kind(::_DenseLowerTri)=:lower
_mf_kind(B::_LABidiag)=B.uplo=='U' ? :upper : :lower
_mf_kind(::_LA.Hermitian)=:hermitian
_mf_kind(A::_LA.Symmetric)=eltype(A)<:Real ? :hermitian : :symmetric
_mf_kind(A::_LASymTridiag)=eltype(A)<:Real ? :hermitian : :symmetric
_mf_kind(A::Union{BandedMatrix,PackedMatrix})=A.kind===:triangular ? (A.upper ? :upper : :lower) :
    A.kind===:hermitian || (A.kind===:symmetric && eltype(A)<:Real) ? :hermitian :
    A.kind===:symmetric ? :symmetric : :general
_mf_kind(::AbstractMatrix)=:general

# log/sqrt of a Hermitian (or real symmetric) A go through the Hermitian
# eigendecomposition, X = V·f(λ)·Vᴴ, which is LinearAlgebra's algorithm.
# This is needed for correctness: the Schur engine gives the negative
# eigenvalues imaginary parts of about ±1e-17 whose sign comes from
# rounding, so sqrt/log picked the branch at random (sqrt(-x-0.0im) = -i√x).
# With λ exactly real, the convention is deterministic: f(complex(λ)), so
# +i√|λ| and log|λ|+iπ. The path is also cheaper and more accurate than the
# complex Schur form.
#   - sqrt: if every λ ≥ -n·eps·max|λ| (semidefinite), the negative λ are
#     clamped to 0 and the result is Hermitian (real Symmetric for real
#     input); otherwise V·sqrt(complex(λ))·Vᴴ, a full complex matrix.
#   - log: every λ > 0 gives a Hermitian result; a zero λ raises
#     DomainError (no finite logarithm); otherwise V·log(complex(λ))·Vᴴ.
# V comes from the structured engine when there is one (eigen_hermitian,
# eigen_tridiagonal); otherwise from a dense Hermitian copy (band/packed
# Hermitian, complex structured wrappers), whose result is dense anyway.
_mf_eigen(A::Union{_LA.Hermitian{<:Any,<:StridedMatrix},_LA.Symmetric{<:Real,<:StridedMatrix}})=(E=_eigen_herm(A;vectors=true); (E.values,E.vectors))
_mf_eigen(A::Union{_LASymTridiag{<:Real},_RealStructHS})=(E=eigen(A); (E.values,E.vectors))
_mf_eigen(A)=(E=_eigen_herm(_LA.Hermitian(Matrix(A));vectors=true); (E.values,E.vectors))
function _mf_hermitian(A,f::Symbol)
    λ,V=_mf_eigen(A); R=real(eltype(λ))
    tol=isempty(λ) ? zero(R) : size(A,1)*eps(R)*maximum(abs,λ)
    if f===:sqrt
        hermitian=all(x->x>=-tol,λ)
        μ=hermitian ? sqrt.(max.(λ,zero(R))) : sqrt.(complex.(λ))
    else
        any(iszero,λ) && throw(DomainError(A,"matrix logarithm requires nonsingular input"))
        hermitian=all(>(zero(R)),λ)
        μ=hermitian ? log.(λ) : log.(complex.(λ))
    end
    S=promote_type(eltype(V),eltype(μ))
    W=Matrix{S}(V); X=_mm(W .* transpose(Vector{S}(μ)),Matrix{S}(adjoint(W)))
    hermitian || return X
    eltype(X)<:Real ? _LA.Symmetric(X) : _LA.Hermitian(X)
end
_mf_posdiag(A)=all(i->real(A[i,i])>0 && iszero(imag(A[i,i])),1:size(A,1))

function _mf_wrap(A,R,f::Symbol)
    k=_mf_kind(A); realin=eltype(A)<:Real
    if f in (:log,:sqrt)   # (the Hermitian kinds do not get here: _mf_hermitian)
        realvalued=realin && k in (:upper,:lower) && _mf_posdiag(A)
    else
        realvalued=realin   # exp, sin, cos of a real matrix are real
    end
    X=realvalued && eltype(R)<:Complex ? real.(R) : R
    k===:upper && return _LA.UpperTriangular(X)
    k===:lower && return _LA.LowerTriangular(X)
    k===:hermitian && return eltype(X)<:Real ? _LA.Symmetric(X) : _LA.Hermitian(X)
    k===:symmetric && return _LA.Symmetric(X)
    X
end

_mf_input(A::_DenseIn)=eltype(A)<:GenericFloat ? A : _dense_input(A)
_mf_input(A::Union{_LABidiag,_LATridiag,_LASymTridiag,_StructHS})=eltype(A)<:GenericFloat ? A : float.(A)
_mf_input(A::BandedMatrix)=eltype(A)<:GenericFloat ? A : _float_band(A)
_mf_input(A::PackedMatrix)=eltype(A)<:GenericFloat ? A : _float_packed(A)
const _MatFunIn=Union{_DenseIn,_LABidiag,_LATridiag,_LASymTridiag,BandedMatrix,PackedMatrix,_StructHS}
for (f,g) in ((:exp,:matrix_exp),(:log,:matrix_log),(:sqrt,:matrix_sqrt),(:sin,:matrix_sin),(:cos,:matrix_cos))
    if f in (:log,:sqrt)
        @eval $f(A::_MatFunIn)=(X=_mf_input(A); _mf_kind(X)===:hermitian ? _mf_hermitian(X,$(QuoteNode(f))) : _mf_wrap(X,$g(X),$(QuoteNode(f))))
    else
        @eval $f(A::_MatFunIn)=(X=_mf_input(A); _mf_wrap(X,$g(X),$(QuoteNode(f))))
    end
    # Diagonal: entry by entry, so the structure is kept (O(n)), as in
    # LinearAlgebra.
    @eval $f(D::_LADiag)=_LA.Diagonal(map(Base.$f,_fvec(D.diag)))
end
