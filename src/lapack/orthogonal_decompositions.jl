# SPDX-License-Identifier: Apache-2.0
struct OrthogonalFactor{T}
    triangular::Matrix{T}
    orthogonal::Matrix{T}
    kind::Symbol
end
function Base.getproperty(F::OrthogonalFactor,s::Symbol)
    s===:Q && return getfield(F,:orthogonal)
    s in (:L,:R) && return getfield(F,:triangular)
    getfield(F,s)
end
"""Householder LQ: A=L*Q, using native QR of A'."""
function lq_factor(A::AbstractMatrix{T};kwargs...) where {T<:GenericFloat}
    F=qr_factor(adjoint(A);kwargs...)
    OrthogonalFactor(Matrix(adjoint(F.R)),Matrix(adjoint(F.Q)),:LQ)
end
"""Householder QL: A=Q*L, lower trapezoid anchored at the lower right."""
function ql_factor(A::AbstractMatrix{T};kwargs...) where {T<:GenericFloat}
    Base.require_one_based_indexing(A)
    F=qr_factor(reverse(A;dims=(1,2));kwargs...)
    OrthogonalFactor(reverse(F.R;dims=(1,2)),reverse(F.Q;dims=(1,2)),:QL)
end
"""Householder RQ: A=R*Q, upper trapezoid anchored at the upper right."""
function rq_factor(A::AbstractMatrix{T};kwargs...) where {T<:GenericFloat}
    F=ql_factor(adjoint(A);kwargs...)
    OrthogonalFactor(Matrix(adjoint(F.L)),Matrix(adjoint(F.Q)),:RQ)
end

struct CompleteOrthogonalFactor{T}
    Q::Matrix{T}
    triangular::Matrix{T}
    Z::Matrix{T}
    p::Vector{Int}
    rank::Int
    shape::Tuple{Int,Int}
end
"""Rank-revealing complete orthogonal decomposition:
A[:,p]=Q[:,1:r]*triangular*Z[1:r,:]. The middle factor is lower triangular.
Uses pivoted QR followed by LQ, and supports minimum-norm least squares.
"""
function complete_orthogonal(A::AbstractMatrix{T};rtol=eps(real(T))*max(size(A)...),threaded=true) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    rtol>=0 || throw(ArgumentError("negative rank tolerance"))
    F=qr_pivoted(A;threaded); m,n=size(A); limit=min(m,n)
    scale=maximum((abs(F.factors[i,i]) for i in 1:limit);init=zero(real(T))); rank=0
    for i in 1:limit
        abs(F.factors[i,i])>rtol*scale || break
        rank=i
    end
    LQ=lq_factor(view(F.R,1:rank,:);threaded)
    CompleteOrthogonalFactor(F.Q,LQ.L[:,1:rank],LQ.Q,F.p,rank,(m,n))
end
function solve(F::CompleteOrthogonalFactor,b::AbstractVecOrMat{T}) where T
    m,n=F.shape; Base.require_one_based_indexing(b)
    size(b,1)==m || throw(DimensionMismatch("complete orthogonal solve dimensions differ"))
    B=ndims(b)==1 ? reshape(b,m,1) : b; r=F.rank
    Y=Matrix{T}(undef,r,size(B,2)); gemm!(Y,adjoint(view(F.Q,:,1:r)),B)
    trsm!(Y,F.triangular;uplo=:L)
    X=Matrix{T}(undef,n,size(B,2)); gemm!(X,adjoint(view(F.Z,1:r,:)),Y)
    result=similar(X)
    for j in axes(X,2), i in 1:n
        result[F.p[i],j]=X[i,j]
    end
    ndims(b)==1 ? vec(result) : result
end
least_squares(F::CompleteOrthogonalFactor,b::AbstractVecOrMat)=solve(F,b)

"""Apply a stored Householder Q on the left/right, with N/T/C semantics.
Aliased factor storage is snapshotted before changing the output.
"""
function apply_q!(B::AbstractMatrix{T},F::QRFactor{T};side=:L,trans=:N,threaded=true,parallel_threshold=262144,blocksize=nothing) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    Base.require_one_based_indexing(B); _check_output_layout(B)
    left=side===:L; side in (:L,:R) || throw(ArgumentError("side must be L or R"))
    tr=_trans_kind(trans); m=size(F.factors,1)
    blocksize=_qr_apply_blocksize(m,blocksize)
    blocksize>0 || throw(ArgumentError("blocksize must be positive"))
    (left ? size(B,1) : size(B,2))==m || throw(DimensionMismatch("Q application dimensions differ"))
    ff=(Base.mightalias(B,F.factors) || Base.mightalias(B,F.tau)) ? QRFactor(copy(F.factors),copy(F.tau)) : F
    conjugated=tr==1 && T<:Complex
    conjugated && (B .= conj.(B))
    adj=tr!=0; forward=left ? adj : !adj; q=length(ff.tau)
    rhs=left ? size(B,2) : size(B,1)
    if rhs>=_QR_BLOCK_MIN_RHS && q>1 && m>=_QR_BLOCK_MIN_ROWS
        _apply_q_blocked!(B,ff,left,adj,forward,blocksize,threaded,1:q,1:rhs)
    else
        _apply_q_unblocked!(B,ff,left,forward,threaded,parallel_threshold)
    end
    conjugated && (B .= conj.(B))
    B
end
# Below this many right-hand sides, one reflector at a time (BLAS 2) does
# the same work without the V copy and T factor of each block.
const _QR_BLOCK_MIN_RHS=8
# Below 96 rows the V copies and T factors cost more than they save (0.8x
# the unblocked application at 64x64); 16-reflector blocks up to 128 rows
# (at 256 they were 1.1x slower than 32).
const _QR_BLOCK_MIN_ROWS=96
_qr_apply_blocksize(m,blocksize)=something(blocksize,m<=128 ? 16 : 32)
# Blocked application (LAPACK ormqr/unmqr): reflector blocks of `blocksize`
# in compact WY form, each applied with two GEMMs and a TRMM; one reflector at
# a time ran at BLAS 2 speed. `cols` restricts the left product to columns
# (qr_q skips the identity columns a block cannot change).
function _apply_q_blocked!(B,F,left,adj,forward,blocksize,threaded,reflectors,cols)
    T=eltype(B); a=F.factors; m=size(a,1); q=last(reflectors)
    nb=min(blocksize,q); rhs=left ? size(B,2) : size(B,1)
    starts=collect(first(reflectors):nb:q); order=forward ? starts : reverse(starts)
    workers=threaded ? _gemm_workers(T,2.0*m*q*rhs,true,8_000_000,min(rhs,Threads.nthreads())) : 1
    if workers<=1
        V=Matrix{T}(undef,m,nb); Tm=Matrix{T}(undef,nb,nb); G=Matrix{T}(undef,nb,nb)
        W=left ? Matrix{T}(undef,nb,rhs) : Matrix{T}(undef,rhs,nb)
        for j0 in order
            p=min(nb,q-j0+1)
            Vv=_qr_block_v!(V,a,j0,p); Tv=_qr_tfactor!(view(Tm,1:p,1:p),Vv,F.tau,j0,view(G,1:p,1:p);threaded)
            target=left ? view(B,j0:m,cols isa Function ? cols(j0) : cols) : view(B,:,j0:m)
            isempty(target) || _qr_block_apply!(target,Vv,Tv,W,left,adj;threaded)
        end
        return B
    end
    # Parallel: all V and T once, then each task applies every block, in
    # order and without nested tasks, to its own slice of B's columns (left)
    # or rows (right). Relying on the threads inside each block's small GEMMs
    # (k=nb) scaled 1.5-2.6x and fell below the former column-split
    # unblocked code at 12-18 threads.
    Vall=Matrix{T}(undef,m,q); Tall=Matrix{T}(undef,nb,q); G=Matrix{T}(undef,nb,nb)
    for j0 in starts
        p=min(nb,q-j0+1)
        Vv=_qr_block_v!(view(Vall,j0:m,j0:j0+p-1),a,j0,p)
        _qr_tfactor!(view(Tall,1:p,j0:j0+p-1),Vv,F.tau,j0,view(G,1:p,1:p);threaded)
    end
    _l1_foreach(rhs,workers) do range,_
        # Task-local names: the sequential branch above assigns W, p, Vv and
        # Tv, which would otherwise make them shared captured variables.
        local Wt=left ? Matrix{T}(undef,nb,length(range)) : Matrix{T}(undef,length(range),nb)
        for j0 in order
            local pt=min(nb,q-j0+1)
            local Vt=view(Vall,j0:m,j0:j0+pt-1)
            local Tt=view(Tall,1:pt,j0:j0+pt-1)
            if left
                local ct=cols isa Function ? cols(j0) : cols
                ct=max(first(ct),first(range)):min(last(ct),last(range))
                isempty(ct) || _qr_block_apply!(view(B,j0:m,ct),Vt,Tt,Wt,true,adj;threaded=false)
            else
                _qr_block_apply!(view(B,range,j0:m),Vt,Tt,Wt,false,adj;threaded=false)
            end
        end
    end
    B
end
function _apply_q_unblocked!(B,ff,left,forward,threaded,parallel_threshold)
    T=eltype(B); m=size(ff.factors,1); q=length(ff.tau)
    rhs=left ? size(B,2) : size(B,1)
    workers=_operation_workers(rhs,Float64(m)*q*rhs,threaded,parallel_threshold)
    _foreach_range(rhs,workers) do range,_
        out=left ? view(B,:,range) : view(B,range,:)
        v=Vector{T}(undef,m); work=Vector{T}(undef,size(out,1))
        for j in (forward ? (1:q) : (q:-1:1))
            tau=ff.tau[j]
            if left && T<:NativeFloat && out isa StridedMatrix && ff.factors isa StridedMatrix
                # Contiguous columns: SIMD Level 1 dot/axpy as in the QR panel.
                iszero(tau) && continue
                vj=view(ff.factors,j+1:m,j)
                @inbounds for col in axes(out,2)
                    c=view(out,j+1:m,col)
                    s=(out[j,col]+dotc(vj,c))*tau
                    out[j,col]-=s
                    axpy!(c,vj,-s)
                end
                continue
            end
            vv=view(v,1:m-j+1); vv[1]=one(T)
            for i in j+1:m
                vv[i-j+1]=ff.factors[i,j]
            end
            left ? _reflect_left!(out,vv,tau,j,axes(out,2)) : _reflect_right!(out,vv,tau,j,axes(out,1),work)
        end
    end
    B
end
apply_q!(B,F::PivotedQRFactor;kwargs...)=apply_q!(B,F.qr;kwargs...)

"""Materialize selected leading columns of Q without allocating the full Q."""
function qr_q(F::QRFactor{T};columns=min(size(F.factors)...),threaded=true,blocksize=nothing) where T
    m=size(F.factors,1); 0<=columns<=m || throw(ArgumentError("invalid Q column count"))
    blocksize=_qr_apply_blocksize(m,blocksize)
    Q=zeros(T,m,columns)
    for i in 1:columns
        Q[i,i]=one(T)
    end
    q=length(F.tau)
    (columns<_QR_BLOCK_MIN_RHS || q<=1 || m<_QR_BLOCK_MIN_ROWS) && return apply_q!(Q,F;threaded)
    # As LAPACK orgqr: applied backwards to [I;0], block j0 cannot change
    # columns < j0 (still unit vectors with zeros in rows >= j0), and
    # reflectors beyond `columns` never act on the first `columns` columns.
    threaded=threaded && generic_threadsafe(T)
    _apply_q_blocked!(Q,F,true,false,false,blocksize,threaded,1:min(q,columns),j0->j0:columns)
end
qr_q(F::PivotedQRFactor;kwargs...)=qr_q(F.qr;kwargs...)
