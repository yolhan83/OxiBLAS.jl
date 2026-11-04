# SPDX-License-Identifier: Apache-2.0
# `factors` keeps the storage type of the factored matrix: a Matrix for
# lu_factor (which copies), the caller's matrix or view for the in-place
# _lu_factor! (OxiBLAS.lu!).
struct LUFactor{T,M<:AbstractMatrix{T}}
    factors::M
    pivots::Vector{Int}
    p::Vector{Int}
    info::Int
end

"""Partial-pivoted blocked LU. Trailing updates use this package's GEMM."""
function lu_factor(A::AbstractMatrix{T};blocksize=128,check=true,threaded=true,pivot=:partial) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    pivot===:partial || return _pivot_lu_factor(A,pivot;check,threaded)
    Base.require_one_based_indexing(A)
    _lu_factor!(Matrix(A);blocksize,check,threaded)
end
"""In-place partial-pivoted blocked LU: F (a Matrix or a strided view) is
overwritten by the unit-lower and upper factors, and is the stored `factors`.
The engine of lu_factor, which hands it a copy."""
function _lu_factor!(F::AbstractMatrix{T};blocksize=128,check=true,threaded=true) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    Base.require_one_based_indexing(F)
    blocksize>0 || throw(ArgumentError("blocksize must be positive"))
    m,n=size(F); q=min(m,n)
    pivots=Vector{Int}(undef,q); info=Ref(0)
    # As LAPACK getrf: panels of `blocksize` columns factored recursively
    # (getrf2), their row interchanges applied once per panel to the columns
    # on each side (laswp), then TRSM and GEMM on the trailing matrix.
    # Swapping whole rows at every pivot and a scalar panel loop ran 2-4x
    # behind OpenBLAS (M5 Max).
    # Lookahead (threaded): once panel k is applied to the next panel's
    # columns, the rest of the trailing update runs in a task while this task
    # factors panel k+1, which otherwise left every other thread idle. Its
    # row interchanges are applied afterwards: they permute whole rows, which
    # the trailing GEMM updates independently, so the result is the
    # sequential one.
    lookahead=threaded && Threads.nthreads()>1 && Float64(m)*n*q>=_LU_LOOKAHEAD_WORK
    factored=0
    for firstcol in 1:blocksize:q
        lastcol=min(q,firstcol+blocksize-1)
        factored<firstcol && _lu_panel!(view(F,firstcol:m,firstcol:lastcol),pivots,firstcol,info,check,threaded)
        _lu_swap!(F,pivots,firstcol,lastcol,1:firstcol-1)
        _lu_swap!(F,pivots,firstcol,lastcol,lastcol+1:n)
        lastcol<n || continue
        trsm!(view(F,firstcol:lastcol,lastcol+1:n),view(F,firstcol:lastcol,firstcol:lastcol);uplo=:L,unit=true,threaded)
        lastcol<m || continue
        L21=view(F,lastcol+1:m,firstcol:lastcol)
        nextlast=min(q,lastcol+blocksize)
        if lookahead && nextlast>lastcol && nextlast<n
            gemm!(view(F,lastcol+1:m,lastcol+1:nextlast),L21,view(F,firstcol:lastcol,lastcol+1:nextlast),-one(T),one(T);threaded)
            rest=Threads.@spawn gemm!(view(F,lastcol+1:m,nextlast+1:n),L21,view(F,firstcol:lastcol,nextlast+1:n),-one(T),one(T);threaded)
            try
                _lu_panel!(view(F,lastcol+1:m,lastcol+1:nextlast),pivots,lastcol+1,info,check,false)
            finally
                wait(rest)
            end
            factored=nextlast
        else
            gemm!(view(F,lastcol+1:m,lastcol+1:n),L21,view(F,firstcol:lastcol,lastcol+1:n),-one(T),one(T);threaded)
        end
    end
    perm=collect(1:m)
    for j in 1:q
        p=pivots[j]; perm[j],perm[p]=perm[p],perm[j]
    end
    LUFactor(F,pivots,perm,info[])
end
# Below this m*n*q the panel is short enough that lookahead tasks do not pay.
const _LU_LOOKAHEAD_WORK=2.0^27
# Interchanges of steps j1:j2 (global row indices in pivots) on columns cols
# of F, column by column (contiguous) as LAPACK laswp.
function _lu_swap!(F,pivots,j1,j2,cols)
    @inbounds for col in cols, j in j1:j2
        p=pivots[j]
        p!=j && ((F[j,col],F[p,col])=(F[p,col],F[j,col]))
    end
    F
end
# Largest-modulus row of column j at or below row j. Complex moduli are
# compared squared (no square root); an overflowing square falls back to abs.
# Ties keep the first row, as the previous scan did.
function _lu_pivot(P,j)
    m=size(P,1); p=j
    if eltype(P)<:Complex
        best=abs2(P[j,j])
        @inbounds for i in j+1:m
            v=abs2(P[i,j]); v>best && (best=v; p=i)
        end
        isfinite(best) && return p
        p=j
    end
    best=abs(P[j,j])
    @inbounds for i in j+1:m
        v=abs(P[i,j]); v>best && (best=v; p=i)
    end
    p
end
# Recursive panel LU (LAPACK getrf2) of P, whose row and column 1 are global
# index `offset`; pivots are stored as global row indices. Leaves of up to
# _lu_leaf columns are eliminated column by column: 32 for real types (64x64
# Float64 1.45 -> 0.96x OpenBLAS against 8), 16 for complex ones (32 slowed
# 128-256).
_lu_leaf(::Type{T}) where T=T<:Complex ? 16 : 32
function _lu_panel!(P,pivots,offset,info,check,threaded)
    mp,np=size(P); T=eltype(P)
    if np<=_lu_leaf(T)
        @inbounds for j in 1:min(mp,np)
            p=_lu_pivot(P,j); pivots[offset+j-1]=offset+p-1
            if p!=j
                for col in 1:np
                    P[j,col],P[p,col]=P[p,col],P[j,col]
                end
            end
            d=P[j,j]
            if iszero(d)
                info[]==0 && (info[]=offset+j-1)
                check && throw(SingularException(offset+j-1))
                continue
            end
            for i in j+1:mp
                P[i,j]/=d
            end
            for col in j+1:np
                u=P[j,col]
                @simd for i in j+1:mp
                    P[i,col]=muladd(-P[i,j],u,P[i,col])
                end
            end
        end
        return P
    end
    h=np÷2
    _lu_panel!(view(P,:,1:h),pivots,offset,info,check,threaded)
    _lu_swap!(P,_LocalPivots(pivots,offset),1,h,h+1:np)
    trsm!(view(P,1:h,h+1:np),view(P,1:h,1:h);uplo=:L,unit=true,threaded)
    mp>h && gemm!(view(P,h+1:mp,h+1:np),view(P,h+1:mp,1:h),view(P,1:h,h+1:np),-one(T),one(T);threaded)
    if mp>h
        _lu_panel!(view(P,h+1:mp,h+1:np),pivots,offset+h,info,check,threaded)
        _lu_swap!(P,_LocalPivots(pivots,offset),h+1,min(mp,np),1:h)
    end
    P
end
# Global pivot indices seen in a view whose first row is global row `offset`.
struct _LocalPivots
    pivots::Vector{Int}
    offset::Int
end
Base.getindex(p::_LocalPivots,j)=p.pivots[p.offset+j-1]-p.offset+1

function Base.getproperty(F::LUFactor,s::Symbol)
    if s===:L
        a=getfield(F,:factors); m,n=size(a); L=zeros(eltype(a),m,min(m,n))
        @inbounds for j in axes(L,2), i in j:m
            L[i,j]=i==j ? one(eltype(a)) : a[i,j]
        end
        return L
    elseif s===:U
        a=getfield(F,:factors); m,n=size(a); U=zeros(eltype(a),min(m,n),n)
        @inbounds for j in 1:n, i in 1:min(j,m)
            U[i,j]=a[i,j]
        end
        return U
    end
    getfield(F,s)
end

function solve(F::LUFactor,b::AbstractVecOrMat)
    a=F.factors; n=_square(a)
    Base.require_one_based_indexing(b)
    size(b,1)==n || throw(DimensionMismatch("LU solve dimensions differ"))
    F.info==0 || throw(SingularException(F.info))
    # A vector right-hand side uses TRSV; a one-column TRSM recursed into
    # tiny GEMM calls.
    if ndims(b)==1
        x=b[F.p]
        trsv!(x,a;uplo=:L,unit=true); trsv!(x,a;uplo=:U)
        return x
    end
    x=b[F.p,:]
    trsm!(x,a;uplo=:L,unit=true)
    trsm!(x,a;uplo=:U)
    x
end

# SingularException/PosDefException: LinearAlgebra's (imported in OxiBLAS.jl).
# `L` keeps the storage type of the factored matrix (see LUFactor).
struct CholeskyFactor{T,M<:AbstractMatrix{T}}
    L::M
end

"""Blocked LLᴴ Cholesky using the lower triangle; GEMM updates native panels."""
function cholesky_factor(A::AbstractMatrix{T};blocksize=nothing,threaded=true) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    n=_square(A)
    # Leaf size of the recursion: 64 for real types (64x64 Float64: 1.6 ->
    # 1.2x OpenBLAS against 32), 32 for complex ones (measured, M5 Max).
    blocksize=something(blocksize,T<:Complex ? 32 : 64)
    blocksize>0 || throw(ArgumentError("blocksize must be positive"))
    # Only the documented lower input triangle is defined/required. This also
    # permits reference-backed scalars with uninitialized upper entries/views.
    L=Matrix{T}(undef,n,n)
    for j in 1:n,i in j:n
        L[i,j]=A[i,j]
    end
    _cholesky_factor!(L;blocksize,threaded)
end
"""In-place Cholesky of the lower triangle of L (a Matrix or a strided view):
L is overwritten by the factor, its strict upper triangle zeroed, and is the
stored `L`. The engine of cholesky_factor, which hands it a copy."""
function _cholesky_factor!(L::AbstractMatrix{T};blocksize=nothing,threaded=true) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    n=_square(L); Base.require_one_based_indexing(L)
    blocksize=something(blocksize,T<:Complex ? 32 : 64)
    blocksize>0 || throw(ArgumentError("blocksize must be positive"))
    _cholesky_recursive!(L,1,blocksize,threaded)
    @inbounds for j in 1:n, i in 1:j-1
        L[i,j]=zero(T)
    end
    CholeskyFactor(L)
end
# Recursive LL' (LAPACK potrf2): L11, then L21=A21*L11^-H (TRSM) and
# A22-=L21*L21' (HERK) before L22, so the large updates run in this
# package's TRSM/HERK. Blocks of up to `leaf` columns are factored right
# looking with contiguous column updates (the former 32-column blocks read
# rows of L in their inner products). `offset` is the global index of row 1.
function _cholesky_recursive!(L,offset,leaf,threaded)
    n=size(L,1); T=eltype(L)
    if n<=leaf
        @inbounds for col in 1:n
            d=real(L[col,col])
            isfinite(d) && d>0 || throw(PosDefException(offset+col-1))
            d=sqrt(d); L[col,col]=d
            for i in col+1:n
                L[i,col]/=d
            end
            for j in col+1:n
                c=conj(L[j,col])
                @simd for i in j:n
                    L[i,j]=muladd(-L[i,col],c,L[i,j])
                end
            end
        end
        return L
    end
    h=n÷2
    _cholesky_recursive!(view(L,1:h,1:h),offset,leaf,threaded)
    panel=view(L,h+1:n,1:h)
    trsm!(panel,view(L,1:h,1:h);side=:R,uplo=:L,trans=:C,threaded)
    herk!(view(L,h+1:n,h+1:n),panel,-one(real(T)),one(real(T));uplo=:L,threaded)
    _cholesky_recursive!(view(L,h+1:n,h+1:n),offset+h,leaf,threaded)
    L
end

function solve(F::CholeskyFactor,b::AbstractVecOrMat)
    Base.require_one_based_indexing(b)
    n=size(F.L,1)
    size(b,1)==n || throw(DimensionMismatch("Cholesky solve dimensions differ"))
    if ndims(b)==1
        x=copy(b); trsv!(x,F.L;uplo=:L); trsv!(x,F.L;uplo=:L,trans=:C)
        return x
    end
    x=copy(b)
    trsm!(x,F.L;uplo=:L)
    trsm!(x,F.L;uplo=:L,trans=:C)
    x
end

struct QRFactor{T}
    factors::Matrix{T}
    tau::Vector{T}
end

"""Compact Householder QR for real/complex rectangular matrices (unpivoted)."""
function qr_factor(A::AbstractMatrix{T};blocksize=nothing,threaded=true,pivot=false) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    pivot && return qr_pivoted(A;threaded)
    return _qr_blocked(A;blocksize=something(blocksize,_qr_blocksize(size(A)...)),threaded)
end
# Measured (M5 Max, Float64/ComplexF32): unblocked when at most 64 columns
# are factored and none trail (0.9x OpenBLAS at 64x64); 16-column panels up
# to 128 (wide 64x128: 1.2x, against 2x unblocked), 32 to 512, 64 beyond
# (1.2 -> 1.05x at 2048).
function _qr_blocksize(m,n)
    q=min(m,n)
    q<=64 && n<=q && return max(q,1)
    q<=128 ? 16 : q<=512 ? 32 : 64
end

# Preserved scalar implementation for comparison and small-panel reference.
function _qr_unblocked(A::AbstractMatrix{T}) where {T<:GenericFloat}
    Base.require_one_based_indexing(A)
    F=Matrix(A); m,n=size(F); tau=zeros(T,min(m,n))
    @inbounds for j in 1:min(m,n)
        r=nrm2(view(F,j:m,j))
        iszero(r) && continue
        phase=iszero(F[j,j]) ? one(T) : F[j,j]/abs(F[j,j])
        den=F[j,j]/r+phase
        τ=one(real(T))+abs(F[j,j])/r
        for i in j+1:m
            F[i,j]=(F[i,j]/r)/den
        end
        F[j,j]=-phase*r
        tau[j]=τ
        for col in j+1:n
            s=F[j,col]
            for i in j+1:m
                s+=conj(F[i,j])*F[i,col]
            end
            s*=τ
            F[j,col]-=s
            for i in j+1:m
                F[i,col]-=F[i,j]*s
            end
        end
    end
    QRFactor(F,tau)
end

function _apply_q!(B,F::QRFactor,adj::Bool)
    a=F.factors; m=size(a,1); q=length(F.tau)
    @inbounds for j in (adj ? (1:q) : (q:-1:1))
        τ=F.tau[j]
        iszero(τ) && continue
        for col in axes(B,2)
            s=B[j,col]
            for i in j+1:m
                s+=conj(a[i,j])*B[i,col]
            end
            s*=τ
            B[j,col]-=s
            for i in j+1:m
                B[i,col]-=a[i,j]*s
            end
        end
    end
    B
end

function Base.getproperty(F::QRFactor,s::Symbol)
    a=getfield(F,:factors); m,n=size(a); T=eltype(a)
    if s===:Q
        return qr_q(F;columns=m)
    elseif s===:R
        R=zeros(T,m,n)
        @inbounds for j in 1:n, i in 1:min(m,j)
            R[i,j]=a[i,j]
        end
        return R
    end
    getfield(F,s)
end

function least_squares(F::QRFactor,b::AbstractVecOrMat;rtol=eps(real(eltype(F.factors)))*max(size(F.factors)...))
    a=F.factors; m,n=size(a)
    m>=n || throw(ArgumentError("underdetermined least squares is not implemented"))
    Base.require_one_based_indexing(b)
    size(b,1)==m || throw(DimensionMismatch("QR solve dimensions differ"))
    scale=maximum(abs(a[i,i]) for i in 1:n;init=zero(real(eltype(a))))
    for i in 1:n
        abs(a[i,i])>rtol*scale || throw(SingularException(i))
    end
    x=copy(b)
    apply_q!(ndims(b)==1 ? reshape(x,m,1) : x,F;trans=:C)
    if ndims(b)==1
        result=x[1:n]; trsv!(result,view(a,1:n,1:n);uplo=:U)
        return result
    end
    result=copy(view(x,1:n,:))
    trsm!(result,view(a,1:n,1:n);uplo=:U)
    result
end
least_squares(A::AbstractMatrix,b::AbstractVecOrMat;kwargs...)=least_squares(qr_factor(A),b;kwargs...)
solve(F::QRFactor,b::AbstractVecOrMat)=least_squares(F,b)

"""Native mixed refinement with Float32 low factors and Float64 residuals.
See mixed_solve's extended implementation for diagnostics and family selection.
"""
function mixed_solve(A::AbstractMatrix{T},b::AbstractVecOrMat{T};maxiter=10,rtol=1e-13,kwargs...) where {T<:Union{Float64,ComplexF64}}
    _mixed_precision_impl(A,b;maxiter,rtol,kwargs...)
end
