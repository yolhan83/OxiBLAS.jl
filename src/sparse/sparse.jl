# SPDX-License-Identifier: Apache-2.0
# Marker for the storage-sharing inner constructors: they wrap vectors that
# already satisfy the compressed invariants (a SparseMatrixCSC, the CSR view
# of its transpose) without validation or copy.
struct _Shared end
struct CSRMatrix{T,I<:Integer} <: SparseArrays.AbstractSparseMatrix{T,I}
    m::Int
    n::Int
    rowptr::Vector{I}
    colind::Vector{I}
    values::Vector{T}
    function CSRMatrix(m::Integer,n::Integer,rowptr::AbstractVector{I},colind::AbstractVector{I},values::AbstractVector{T}) where {I<:Integer,T<:GenericFloat}
        isconcretetype(T) || throw(ArgumentError("a concrete sparse scalar type is required"))
        Base.require_one_based_indexing(rowptr,colind,values)
        min(m,n)>=0 || throw(ArgumentError("negative sparse dimension"))
        length(rowptr)==m+1 && length(colind)==length(values) || throw(DimensionMismatch("invalid CSR array lengths"))
        rowptr[1]==1 && rowptr[end]==length(values)+1 || throw(ArgumentError("invalid CSR endpoints"))
        issorted(rowptr) || throw(ArgumentError("CSR row pointers must be monotonic"))
        # Duplicate check without a Set per row (6-8 allocations each): a row
        # with strictly increasing columns, the usual case, has no duplicate;
        # the other rows share one marker array, allocated at the first one.
        mark=nothing
        for i in 1:m
            lo,hi=rowptr[i],rowptr[i+1]-1; increasing=true
            for p in lo:hi
                1<=colind[p]<=n || throw(ArgumentError("CSR column out of range"))
                p>lo && colind[p]<=colind[p-1] && (increasing=false)
            end
            increasing && continue
            mark===nothing && (mark=zeros(Int,n))
            for p in lo:hi
                j=Int(colind[p]); mark[j]==i && throw(ArgumentError("duplicate CSR entry")); mark[j]=i
            end
        end
        new{T,I}(m,n,copy(rowptr),copy(colind),copy(values))
    end
    CSRMatrix(::_Shared,m::Integer,n::Integer,rowptr::Vector{I},colind::Vector{I},values::Vector{T}) where {I<:Integer,T<:GenericFloat}=
        new{T,I}(m,n,rowptr,colind,values)
end
Base.size(A::CSRMatrix)=(A.m,A.n)
Base.dataids(A::CSRMatrix)=(Base.dataids(A.rowptr)...,Base.dataids(A.colind)...,Base.dataids(A.values)...)
@inline function Base.getindex(A::CSRMatrix{T},i::Int,j::Int) where T
    @boundscheck checkbounds(A,i,j)
    for p in A.rowptr[i]:A.rowptr[i+1]-1
        A.colind[p]==j && return A.values[p]
    end
    zero(T)
end
function sparse_mul!(y::AbstractVector{T},A::CSRMatrix{T},x::AbstractVector{T},α=one(T),β=zero(T);trans=:N,threaded=true,parallel_threshold=262144) where T
    threaded=threaded && generic_threadsafe(T)
    AA=_op(A,trans)
    size(AA)==(length(y),length(x)) || throw(DimensionMismatch("sparse multiply dimensions differ"))
    Base.require_one_based_indexing(y,x)
    _check_vector_output(y)
    parallel_threshold>=0 || throw(ArgumentError("negative parallel threshold"))
    a,b=convert(T,α),convert(T,β)    # new names: a reassigned argument captured by the closures below would be boxed
    (Base.mightalias(y,A.values) || Base.mightalias(y,A.rowptr) || Base.mightalias(y,A.colind)) && throw(ArgumentError("output must not alias CSR storage"))
    (iszero(a) || isempty(x)) && return _scale!(y,b)
    xx=Base.mightalias(y,x) ? copy(x) : x
    _scale!(y,b)
    if trans in (:N,'N',:n,'n')
        workers=_operation_workers(A.m,length(A.values),threaded,parallel_threshold)
        _foreach_range(A.m,workers) do rows,_
            @inbounds for i in rows
                s=zero(T)
                for p in A.rowptr[i]:A.rowptr[i+1]-1
                    s+=A.values[p]*xx[A.colind[p]]
                end
                y[i]+=a*s
            end
        end
    else
        conjugate=trans in (:C,'C',:c,'c')
        workers=_operation_workers(A.n,length(A.values),threaded,parallel_threshold)
        if workers>1
            C=CSCMatrix(A)
            _foreach_range(A.n,workers) do columns,_
                @inbounds for j in columns
                    s=zero(T)
                    for p in C.colptr[j]:C.colptr[j+1]-1
                        s+=(conjugate ? conj(C.values[p]) : C.values[p])*xx[C.rowind[p]]
                    end
                    y[j]+=a*s
                end
            end
        else
            @inbounds for i in 1:A.m, p in A.rowptr[i]:A.rowptr[i+1]-1
                y[A.colind[p]]+=a*(conjugate ? conj(A.values[p]) : A.values[p])*xx[i]
            end
        end
    end
    y
end
_native_mv!(y,A,x)=gemv!(y,A,x)
_native_mv!(y,A::CSRMatrix,x)=sparse_mul!(y,A,x)
function _precondition!(z,r,diagonal)
    if diagonal===nothing
        copyto!(z,r)
    else
        @. z=r/diagonal
    end
    z
end

"""
    cg(A::AbstractMatrix{T}, b::AbstractVector{T}; x0=zeros(T,length(b)), rtol=sqrt(eps(real(T))), atol=zero(real(T)), maxiter=2length(b), diagonal=nothing) -> NamedTuple

Conjugate gradient for Hermitian positive definite matrices; optional positive
real Jacobi diagonal. Reports the true residual and a termination reason.
"""
function cg(A::AbstractMatrix{T},b::AbstractVector{T};x0=zeros(T,length(b)),
            rtol=sqrt(eps(real(T))),atol=zero(real(T)),maxiter=2length(b),diagonal=nothing) where {T<:NativeFloat}
    n=_square(A)
    length(b)==length(x0)==n || throw(DimensionMismatch("CG dimensions differ"))
    min(rtol,atol,maxiter)>=0 || throw(ArgumentError("invalid CG tolerance/iteration limit"))
    Base.require_one_based_indexing(b,x0)
    if diagonal!==nothing
        Base.require_one_based_indexing(diagonal)
        length(diagonal)==n && all(d->isreal(d) && isfinite(d) && real(d)>0,diagonal) || throw(ArgumentError("Jacobi diagonal must be finite and positive"))
    end
    x=copy(x0); r=similar(b); z=similar(b); p=similar(b); Ap=similar(b)
    _native_mv!(Ap,A,x)
    @. r=b-Ap
    target=max(atol,rtol*nrm2(b))
    residual=nrm2(r)
    residual<=target && return (;x,converged=true,iterations=0,residual,reason=:converged)
    _precondition!(z,r,diagonal)
    copyto!(p,z); rho=real(dotc(r,z))
    for iter in 1:maxiter
        _native_mv!(Ap,A,p)
        denom=real(dotc(p,Ap))
        if !isfinite(denom) || denom<=0 || !isfinite(rho) || rho<=0
            return (;x,converged=false,iterations=iter-1,residual,reason=:breakdown)
        end
        α=rho/denom
        axpy!(x,p,α); axpy!(r,Ap,-α)
        residual=nrm2(r)
        if residual<=target || iter==maxiter
            _native_mv!(Ap,A,x)
            @. r=b-Ap
            residual=nrm2(r)
            residual<=target && return (;x,converged=true,iterations=iter,residual,reason=:converged)
            # Residual replacement must restart the search direction.
            _precondition!(z,r,diagonal)
            copyto!(p,z); rho=real(dotc(r,z))
            continue
        end
        _precondition!(z,r,diagonal)
        next_rho=real(dotc(r,z))
        β=next_rho/rho
        @. p=z+β*p
        rho=next_rho
    end
    # maxiter=0 also reports a true residual.
    (;x,converged=false,iterations=maxiter,residual,reason=:iteration_limit)
end
