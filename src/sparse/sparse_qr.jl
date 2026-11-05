# SPDX-License-Identifier: Apache-2.0
struct SparseQRFactor{T,R}
    R::CSRMatrix{T,Int}
    reflector_indices::Vector{Vector{Int}}
    reflector_values::Vector{Vector{T}}
    tau::Vector{R}
    p::Vector{Int}
    shape::Tuple{Int,Int}
end
"""Sparse Householder QR, A[:,p] = Q·[R; 0].

`pivot=true` (default): rank-revealing column pivoting by recomputed norms,
column by column, with fill in column dictionaries. `pivot=false`:
multifrontal QR (src/sparse/sparse_qr_multifrontal.jl) in the column `ordering`:
`:natural` (p = 1:n), `:amd` (AMD of AᵀA), `:colamd` (COLAMD, SPQR's default
kind of order) or a permutation vector. The input is never expanded densely.
"""
function sparse_qr(A::NativeSparse;pivot=true,ordering=:natural)
    m,n=size(A)
    if !pivot
        q=ordering isa AbstractVector ? _checked_sparse_permutation(ordering,n) :
          ordering===:natural ? collect(1:n) :
          ordering===:amd ? amd_ordering(A;kind=:ata) : ordering===:colamd ? colamd_ordering(A) :
          throw(ArgumentError("unknown sparse QR ordering"))
        return _qr_multifrontal(A,q)
    end
    ordering===:natural || throw(ArgumentError("column pivoting fixes the order: use pivot=false with an ordering"))
    C=CSRMatrix(A); T=eltype(C); R=real(T)
    columns=[Dict{Int,T}() for _ in 1:n]
    for i in 1:m, p in C.rowptr[i]:C.rowptr[i+1]-1
        columns[C.colind[p]][i]=C.values[p]
    end
    indices=Vector{Vector{Int}}(); vectors=Vector{Vector{T}}(); tau=R[]; perm=collect(1:n); scratch=zeros(T,m)
    for j in 1:min(m,n)
        if pivot
            col=j; best=zero(R)
            for k in j:n
                fill!(scratch,zero(T))
                for (i,value) in columns[k]
                    i>=j && (scratch[i]=value)
                end
                norm=nrm2(view(scratch,j:m)); norm>best && (best=norm; col=k)
            end
            if col!=j
                columns[j],columns[col]=columns[col],columns[j]; perm[j],perm[col]=perm[col],perm[j]
            end
        end
        fill!(scratch,zero(T))
        for (i,value) in columns[j]
            i>=j && (scratch[i]=value)
        end
        v=view(scratch,j:m); beta,t=_householder!(v)
        rows=Int[]; values=T[]
        for (offset,value) in enumerate(v)
            if !iszero(value)
                push!(rows,j+offset-1); push!(values,value)
            end
        end
        push!(indices,rows); push!(vectors,values); push!(tau,t)
        if !iszero(t)
            for col in j+1:n
                s=zero(T)
                for (i,value) in zip(rows,values)
                    s+=conj(value)*get(columns[col],i,zero(T))
                end
                s*=t
                for (i,value) in zip(rows,values)
                    updated=get(columns[col],i,zero(T))-value*s
                    iszero(updated) ? delete!(columns[col],i) : (columns[col][i]=updated)
                end
            end
        end
        for i in collect(keys(columns[j]))
            i>j && delete!(columns[j],i)
        end
        columns[j][j]=beta
    end
    rows=[Dict{Int,T}() for _ in 1:m]
    for j in 1:n, (i,value) in columns[j]
        rows[i][j]=value
    end
    SparseQRFactor(_rows_to_csr(rows,T,n),indices,vectors,tau,perm,(m,n))
end
function _reflectors_alias(B,vs)
    for v in vs
        Base.mightalias(B,v) && return true
    end
    false
end
function sparse_apply_q!(B::AbstractMatrix{T},F::SparseQRFactor{T};trans=:N) where T
    Base.require_one_based_indexing(B); _check_output_layout(B)
    size(B,1)==F.shape[1] || throw(DimensionMismatch("sparse Q application dimensions differ"))
    tr=_trans_kind(trans); conjugated=tr==1 && T<:Complex
    # The public factor owns its reflector storage; protect unusual alias views.
    aliased=_reflectors_alias(B,F.reflector_values)
    vectors=aliased ? copy.(F.reflector_values) : F.reflector_values
    conjugated && (B .= conj.(B))
    q=length(F.tau)
    # B -= τ·v·(vᴴB) per reflector: a dot product with four accumulators,
    # then an axpy, on the reflector's rows.
    @inbounds for k in (tr==0 ? (q:-1:1) : (1:q))
        t=F.tau[k]; iszero(t) && continue
        idx=F.reflector_indices[k]; v=vectors[k]; len=length(idx)
        for col in axes(B,2)
            x=view(B,:,col)
            s=_cdot_idx(v,x,idx,1,len)*t
            iszero(s) && continue
            for l in 1:len
                x[idx[l]]=muladd(-v[l],s,x[idx[l]])
            end
        end
    end
    conjugated && (B .= conj.(B))
    B
end
function _qr_solve!(F::SparseQRFactor{T},X::AbstractMatrix{T};rtol=eps(real(T))*max(F.shape...)) where T
    m,n=F.shape; Base.require_one_based_indexing(X)
    m>=n && size(X,1)==m || throw(DimensionMismatch("sparse QR solve requires a tall compatible matrix"))
    scale=zero(real(T))
    for i in 1:n
        scale=max(scale,abs(F.R[i,i]))
    end
    for i in 1:n
        abs(F.R[i,i])>rtol*scale || throw(SingularException(i))
    end
    sparse_apply_q!(X,F;trans=:C)
    # Rows n+1:m of R are empty: _trisolve_kernel! reads only rows 1:n.
    top=view(X,1:n,:)
    _trisolve_kernel!(top,F.R.rowptr,F.R.colind,F.R.values,n,Val(true),Val(false),Val(false),Val(false))
    if F.p!=1:n
        w=Vector{T}(undef,n)
        @inbounds for col in axes(top,2)
            for i in 1:n; w[F.p[i]]=top[i,col]; end
            for i in 1:n; top[i,col]=w[i]; end
        end
    end
    X
end
function solve(F::SparseQRFactor{T},b::AbstractVecOrMat{T};rtol=eps(real(T))*max(F.shape...)) where T
    m,n=F.shape; Base.require_one_based_indexing(b)
    m>=n && size(b,1)==m || throw(DimensionMismatch("sparse QR solve requires a tall compatible matrix"))
    if ndims(b)==1
        X=copy(b)
        _qr_solve!(F,reshape(X,m,1);rtol)
        return m==n ? X : resize!(X,n)
    end
    X=copy(b)
    _qr_solve!(F,X;rtol)
    m==n ? X : X[1:n,:]
end
least_squares(F::SparseQRFactor,b::AbstractVecOrMat;kwargs...)=solve(F,b;kwargs...)

function sparse_svd(A::NativeSparse,rank::Integer;oversampling=8,power_iterations=2,seed=0x0b1a5,threaded=true,kwargs...)
    T=eltype(A); m,n=size(A)
    0<=rank<=min(m,n) && min(oversampling,power_iterations)>=0 || throw(ArgumentError("invalid sparse SVD options"))
    C=CSRMatrix(A); q=min(min(m,n),rank+oversampling); omega=_random_projection(T,n,q,seed)
    Y=Matrix{T}(undef,m,q); sparse_mul!(Y,C,omega;threaded); Q=qr_q(qr_factor(Y;threaded);columns=q)
    for _ in 1:power_iterations
        Z=Matrix{T}(undef,n,q); sparse_mul!(Z,C,Q;trans=:C,threaded)
        W=qr_q(qr_factor(Z;threaded);columns=q); sparse_mul!(Y,C,W;threaded)
        Q=qr_q(qr_factor(Y;threaded);columns=q)
    end
    Z=Matrix{T}(undef,n,q); sparse_mul!(Z,C,Q;trans=:C,threaded)
    F=svd_bidiagonal(adjoint(Z);threaded,kwargs...); U=Matrix{T}(undef,m,rank)
    gemm!(U,Q,view(F.U,:,1:rank);threaded)
    SVDFactor(U,F.S[1:rank],F.Vt[1:rank,:],F.sweeps)
end
