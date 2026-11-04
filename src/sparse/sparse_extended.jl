# SPDX-License-Identifier: Apache-2.0
"""Owned compressed-column sparse matrix with validated 1-based indices.

A `SparseArrays.AbstractSparseMatrix`, like the other OxiBLAS formats (not an
`AbstractSparseMatrixCSC`: operations on OxiBLAS sparse types end in OxiBLAS
kernels, not in inherited SparseArrays ones; see
reports/ARCHITECTURE_CSC_A_vs_B.md). Row indices are unique and sorted within
each column, so `SparseMatrixCSC(A)` always shares the storage; explicit stored
zeros are allowed, as in `SparseMatrixCSC`."""
struct CSCMatrix{T,I<:Integer} <: SparseArrays.AbstractSparseMatrix{T,I}
    m::Int
    n::Int
    colptr::Vector{I}
    rowind::Vector{I}
    values::Vector{T}
    function CSCMatrix(m::Integer,n::Integer,colptr::AbstractVector{I},rowind::AbstractVector{I},values::AbstractVector{T}) where {T<:GenericFloat,I<:Integer}
        # Transposed CSR has the pointer/range/uniqueness invariants; the
        # (copied) row indices are then sorted within each column.
        C=CSRMatrix(n,m,colptr,rowind,values)
        rows,vals=C.colind,C.values
        for j in 1:n
            r=C.rowptr[j]:C.rowptr[j+1]-1
            issorted(view(rows,r)) && continue
            o=sortperm(view(rows,r)); rows[r]=rows[r][o]; vals[r]=vals[r][o]
        end
        new{T,I}(m,n,C.rowptr,rows,vals)
    end
    # Unchecked: for storage already valid and sorted (a SparseMatrixCSC, a
    # copy of a CSCMatrix). A transient unsorted wrapper is only ever handed
    # to SparseMatrixCSC(::CSCMatrix), which sorts.
    CSCMatrix(::_Shared,m::Integer,n::Integer,colptr::Vector{I},rowind::Vector{I},values::Vector{T}) where {T<:GenericFloat,I<:Integer}=
        new{T,I}(m,n,colptr,rowind,values)
end
Base.size(A::CSCMatrix)=(A.m,A.n)
Base.dataids(A::CSCMatrix)=(Base.dataids(A.colptr)...,Base.dataids(A.rowind)...,Base.dataids(A.values)...)
function Base.getindex(A::CSCMatrix{T},i::Int,j::Int) where T
    @boundscheck checkbounds(A,i,j)
    for p in A.colptr[j]:A.colptr[j+1]-1
        A.rowind[p]==i && return A.values[p]
    end
    zero(T)
end

"""Canonical coordinate sparse storage. Duplicate coordinates are summed and
exact zero entries are removed during construction.
"""
struct COOMatrix{T,I<:Integer} <: SparseArrays.AbstractSparseMatrix{T,I}
    m::Int
    n::Int
    rows::Vector{I}
    cols::Vector{I}
    values::Vector{T}
    function COOMatrix(m::Integer,n::Integer,rows::AbstractVector{I},cols::AbstractVector{I},values::AbstractVector{T}) where {T<:GenericFloat,I<:Integer}
        isconcretetype(T) || throw(ArgumentError("a concrete sparse scalar type is required"))
        Base.require_one_based_indexing(rows,cols,values)
        min(m,n)>=0 && length(rows)==length(cols)==length(values) || throw(ArgumentError("invalid COO dimensions"))
        all(i->1<=i<=m,rows) && all(j->1<=j<=n,cols) || throw(ArgumentError("COO index out of range"))
        order=sortperm(eachindex(values);by=p->(rows[p],cols[p]))
        rr=I[]; cc=I[]; vv=T[]
        for p in order
            if !isempty(vv) && rr[end]==rows[p] && cc[end]==cols[p]
                vv[end]+=values[p]
            else
                push!(rr,rows[p]); push!(cc,cols[p]); push!(vv,values[p])
            end
        end
        keep=findall(!iszero,vv)
        new{T,I}(m,n,rr[keep],cc[keep],vv[keep])
    end
end
Base.size(A::COOMatrix)=(A.m,A.n)
Base.dataids(A::COOMatrix)=(Base.dataids(A.rows)...,Base.dataids(A.cols)...,Base.dataids(A.values)...)
function Base.getindex(A::COOMatrix{T},i::Int,j::Int) where T
    @boundscheck checkbounds(A,i,j)
    for p in eachindex(A.values)
        A.rows[p]==i && A.cols[p]==j && return A.values[p]
    end
    zero(T)
end
function CSRMatrix(A::COOMatrix{T}) where T
    ptr=ones(Int,A.m+1)
    for i in A.rows
        ptr[i+1]+=1
    end
    for i in 1:A.m
        ptr[i+1]+=ptr[i]-1
    end
    CSRMatrix(A.m,A.n,ptr,Int.(A.cols),A.values)
end
function CSCMatrix(A::CSRMatrix{T}) where T
    ptr=ones(Int,A.n+1)
    for j in A.colind
        ptr[j+1]+=1
    end
    for j in 1:A.n
        ptr[j+1]+=ptr[j]-1
    end
    cursor=copy(ptr); rows=Vector{Int}(undef,length(A.values)); vals=Vector{T}(undef,length(A.values))
    for i in 1:A.m, p in A.rowptr[i]:A.rowptr[i+1]-1
        j=A.colind[p]; q=cursor[j]; rows[q]=i; vals[q]=A.values[p]; cursor[j]+=1
    end
    CSCMatrix(A.m,A.n,ptr,rows,vals)
end
function CSRMatrix(A::CSCMatrix{T}) where T
    C=CSCMatrix(CSRMatrix(A.n,A.m,A.colptr,A.rowind,A.values))
    CSRMatrix(A.m,A.n,C.colptr,C.rowind,C.values)
end
CSCMatrix(A::COOMatrix)=CSCMatrix(CSRMatrix(A))
CSRMatrix(A::CSRMatrix)=CSRMatrix(A.m,A.n,A.rowptr,A.colind,A.values)
CSCMatrix(A::CSCMatrix)=CSCMatrix(A.m,A.n,A.colptr,A.rowind,A.values)
function CSRMatrix(A::AbstractMatrix{T};droptol=zero(real(T))) where {T<:GenericFloat}
    Base.require_one_based_indexing(A); droptol>=0 || throw(ArgumentError("negative sparse drop tolerance"))
    m,n=size(A); ptr=Int[1]; cols=Int[]; vals=T[]
    for i in 1:m
        for j in 1:n
            !(abs(A[i,j])<=droptol) && (push!(cols,j); push!(vals,A[i,j]))
        end
        push!(ptr,length(vals)+1)
    end
    CSRMatrix(m,n,ptr,cols,vals)
end
CSCMatrix(A::AbstractMatrix{T};kwargs...) where {T<:GenericFloat}=CSCMatrix(CSRMatrix(A;kwargs...))
function COOMatrix(A::CSRMatrix)
    rows=Int[]
    for i in 1:A.m
        append!(rows,fill(i,Int(A.rowptr[i+1]-A.rowptr[i])))
    end
    COOMatrix(A.m,A.n,rows,Int.(A.colind),A.values)
end
COOMatrix(A::CSCMatrix)=COOMatrix(CSRMatrix(A))

function sparse_mul!(y::AbstractVector,A::Union{CSCMatrix,COOMatrix},x::AbstractVector,α=one(eltype(y)),β=zero(eltype(y));kwargs...)
    Base.mightalias(y,A) && throw(ArgumentError("output must not alias sparse storage"))
    sparse_mul!(y,CSRMatrix(A),x,α,β;kwargs...)
end
# Walk compressed columns directly for serial A*x and for all transposed dots.
# Keep the established CSR conversion for parallel A*x until alternatives are
# measured: scattering the same y from column tasks would introduce races.
function sparse_mul!(y::AbstractVector{T},A::CSCMatrix{T},x::AbstractVector{T},alpha=one(T),beta=zero(T);
                     trans=:N,threaded=true,parallel_threshold=262144) where T
    threaded=threaded && generic_threadsafe(T)
    tr=_trans_kind(trans);Base.require_one_based_indexing(y,x)
    size(_op(A,trans))==(length(y),length(x)) || throw(DimensionMismatch("CSC multiply dimensions differ"))
    _check_vector_output(y);parallel_threshold>=0 || throw(ArgumentError("negative parallel threshold"))
    Base.mightalias(y,A) && throw(ArgumentError("output must not alias CSC storage"))
    alpha,beta=convert(T,alpha),convert(T,beta)
    (iszero(alpha) || isempty(x)) && return _scale!(y,beta)
    workers=_operation_workers(length(y),length(A.values),threaded,parallel_threshold)
    tr==0 && workers>1 && return sparse_mul!(y,CSRMatrix(A),x,alpha,beta;trans,threaded,parallel_threshold)
    xx=Base.mightalias(y,x) ? copy(x) : x
    _scale!(y,beta)
    if tr==0
        @inbounds for j in 1:A.n
            value=alpha*xx[j]
            for p in A.colptr[j]:A.colptr[j+1]-1
                i=A.rowind[p];y[i]=_arithmetic_madd(A.values[p],value,y[i])
            end
        end
    else
        _foreach_range(A.n,workers) do columns,_
            @inbounds for j in columns
                total=zero(T)
                for p in A.colptr[j]:A.colptr[j+1]-1
                    value=tr==2 ? conj(A.values[p]) : A.values[p]
                    total=_arithmetic_madd(value,xx[A.rowind[p]],total)
                end
                y[j]+=alpha*total
            end
        end
    end
    y
end

_native_mv!(y,A::Union{CSCMatrix,COOMatrix},x)=sparse_mul!(y,A,x)
function sparse_mul!(Y::AbstractMatrix{T},A::Union{CSRMatrix{T},CSCMatrix{T},COOMatrix{T}},X::AbstractMatrix{T},α=one(T),β=zero(T);trans=:N,threaded=true,parallel_threshold=262144) where T
    threaded=threaded && generic_threadsafe(T)
    AA=_op(A,trans); Base.require_one_based_indexing(Y,X)
    size(AA,2)==size(X,1) && size(Y)==(size(AA,1),size(X,2)) || throw(DimensionMismatch("sparse matrix product dimensions differ"))
    _check_output_layout(Y)
    Base.mightalias(Y,A) && throw(ArgumentError("output must not alias sparse storage"))
    α,β=convert(T,α),convert(T,β)
    iszero(α) && return _scale!(Y,β)
    XX=Base.mightalias(Y,X) ? copy(X) : X
    C=A isa CSRMatrix ? A : CSRMatrix(A)
    workers=_operation_workers(size(Y,2),Float64(length(C.values))*size(Y,2),threaded,parallel_threshold)
    _foreach_range(size(Y,2),workers) do columns,_
        for j in columns
            sparse_mul!(view(Y,:,j),C,view(XX,:,j),α,β;trans,threaded=false)
        end
    end
    Y
end

"""Gustavson CSR×CSR multiplication with a sparse row accumulator and sorted
output rows. No dense matrix expansion or external sparse backend.
"""
function sparse_matmul(A::CSRMatrix{T},B::CSRMatrix{T};droptol=zero(real(T))) where T
    A.n==B.m || throw(DimensionMismatch("sparse matrix product dimensions differ"))
    droptol>=0 || throw(ArgumentError("negative sparse drop tolerance"))
    ptr=Int[1]; cols=Int[]; vals=T[]; marks=zeros(Int,B.n); sums=zeros(T,B.n); touched=Int[]
    for i in 1:A.m
        empty!(touched)
        for p in A.rowptr[i]:A.rowptr[i+1]-1
            k=A.colind[p]; a=A.values[p]
            for q in B.rowptr[k]:B.rowptr[k+1]-1
                j=B.colind[q]
                if marks[j]!=i
                    marks[j]=i; sums[j]=zero(T); push!(touched,j)
                end
                sums[j]+=a*B.values[q]
            end
        end
        sort!(touched)
        for j in touched
            !(abs(sums[j])<=droptol) && (push!(cols,j); push!(vals,sums[j]))
        end
        push!(ptr,length(vals)+1)
    end
    CSRMatrix(A.m,B.n,ptr,cols,vals)
end
sparse_matmul(A::Union{CSRMatrix,CSCMatrix,COOMatrix},B::Union{CSRMatrix,CSCMatrix,COOMatrix};kwargs...)=sparse_matmul(CSRMatrix(A),CSRMatrix(B);kwargs...)

struct ILUFactor{T}
    factors::CSRMatrix{T,Int}
    diagonal::Vector{Int}
end
"""Zero-fill incomplete LU on the input sparsity pattern. Requires explicit,
nonzero pivots; numerical breakdown is reported rather than silently patched.
"""
function ilu0(A::Union{CSRMatrix{T},CSCMatrix{T},COOMatrix{T}}) where T
    n=_square(A); C=CSRMatrix(COOMatrix(CSRMatrix(A))) # canonical row ordering
    diag=zeros(Int,n); marker=zeros(Int,n)
    for i in 1:n
        for p in C.rowptr[i]:C.rowptr[i+1]-1
            marker[C.colind[p]]=p
            C.colind[p]==i && (diag[i]=p)
        end
        diag[i]!=0 || throw(SingularException(i))
        for p in C.rowptr[i]:diag[i]-1
            j=C.colind[p]; d=C.values[diag[j]]
            iszero(d) && throw(SingularException(j))
            C.values[p]/=d; l=C.values[p]
            for q in diag[j]+1:C.rowptr[j+1]-1
                col=C.colind[q]; r=marker[col]
                r!=0 && (C.values[r]-=l*C.values[q])
            end
        end
        iszero(C.values[diag[i]]) && throw(SingularException(i))
        for p in C.rowptr[i]:C.rowptr[i+1]-1
            marker[C.colind[p]]=0
        end
    end
    ILUFactor(C,diag)
end
function _precondition!(z,r,F::ILUFactor)
    n=size(F.factors,1); length(z)==length(r)==n || throw(DimensionMismatch("ILU dimensions differ"))
    copyto!(z,r); A=F.factors
    for i in 1:n
        for p in A.rowptr[i]:F.diagonal[i]-1
            z[i]-=A.values[p]*z[A.colind[p]]
        end
    end
    for i in n:-1:1
        for p in F.diagonal[i]+1:A.rowptr[i+1]-1
            z[i]-=A.values[p]*z[A.colind[p]]
        end
        z[i]/=A.values[F.diagonal[i]]
    end
    z
end
solve(F::ILUFactor,b::AbstractVector)=_precondition!(copy(b),b,F)

function _krylov_setup(A,b,x0,rtol,atol,maxiter)
    n=_square(A); Base.require_one_based_indexing(b,x0)
    length(b)==length(x0)==n || throw(DimensionMismatch("Krylov solve dimensions differ"))
    min(rtol,atol,maxiter)>=0 || throw(ArgumentError("invalid Krylov options"))
    max(atol,rtol*nrm2(b))
end
function _check_preconditioner(P,n)
    P===nothing && return
    if P isa ILUFactor
        size(P.factors)==(n,n) || throw(DimensionMismatch("preconditioner dimensions differ"))
    elseif P isa AbstractVector
        Base.require_one_based_indexing(P)
        length(P)==n && all(x->isfinite(x) && !iszero(x),P) || throw(ArgumentError("invalid preconditioner diagonal"))
    else
        throw(ArgumentError("unsupported preconditioner"))
    end
end
function _true_residual!(r,A,x,b,temp)
    _native_mv!(temp,A,x)
    for i in eachindex(r)
        r[i]=b[i]-temp[i]
    end
    nrm2(r)
end

"""Restarted, left-preconditioned GMRES with two-pass modified Gram–Schmidt.
Convergence is checked against the true, unpreconditioned residual every restart.
`preconditioner` may be nothing, a diagonal vector, or an ILUFactor.
"""
function gmres(A::AbstractMatrix{T},b::AbstractVector{T};x0=zeros(T,length(b)),restart=min(30,max(1,length(b))),
               maxiter=2length(b),rtol=sqrt(eps(real(T))),atol=zero(real(T)),preconditioner=nothing) where {T<:NativeFloat}
    target=_krylov_setup(A,b,x0,rtol,atol,maxiter)
    _check_preconditioner(preconditioner,length(b))
    restart>0 || throw(ArgumentError("restart must be positive"))
    n=length(b); x=copy(x0); r=similar(b); work=similar(b); z=similar(b)
    residual=_true_residual!(r,A,x,b,work); iterations=0
    residual<=target && return (;x,converged=true,iterations,residual,reason=:converged)
    q=min(restart,n); V=zeros(T,n,q+1); H=zeros(T,q+1,q); g=zeros(T,q+1)
    cs=zeros(real(T),q); sn=zeros(T,q)
    while iterations<maxiter
        _precondition!(z,r,preconditioner); β=nrm2(z)
        (!isfinite(β) || iszero(β)) && return (;x,converged=false,iterations,residual,reason=:breakdown)
        for i in 1:n
            V[i,1]=z[i]/β
        end
        fill!(H,zero(T)); fill!(g,zero(T)); g[1]=β; used=0; breakdown=false
        for j in 1:min(q,maxiter-iterations)
            _native_mv!(work,A,view(V,:,j)); _precondition!(z,work,preconditioner)
            for _ in 1:2, i in 1:j
                h=dotc(view(V,:,i),z); H[i,j]+=h; axpy!(z,view(V,:,i),-h)
            end
            h=nrm2(z); H[j+1,j]=h
            if !iszero(h)
                for i in 1:n
                    V[i,j+1]=z[i]/h
                end
            end
            for i in 1:j-1
                a=H[i,j]; bnext=H[i+1,j]
                H[i,j]=cs[i]*a+sn[i]*bnext; H[i+1,j]=cs[i]*bnext-conj(sn[i])*a
            end
            c,s,rr=rotg(H[j,j],H[j+1,j]); cs[j]=c; sn[j]=s; H[j,j]=rr; H[j+1,j]=zero(T)
            a=g[j]; g[j]=c*a; g[j+1]=-conj(s)*a
            used=j; iterations+=1; breakdown=iszero(h)
            (abs(g[j+1])<=target || breakdown) && break
        end
        y=copy(view(g,1:used))
        if any(iszero(H[i,i]) for i in 1:used)
            return (;x,converged=false,iterations,residual,reason=:breakdown)
        end
        trsv!(y,view(H,1:used,1:used);uplo=:U)
        for j in 1:used
            axpy!(x,view(V,:,j),y[j])
        end
        residual=_true_residual!(r,A,x,b,work)
        residual<=target && return (;x,converged=true,iterations,residual,reason=:converged)
        breakdown && return (;x,converged=false,iterations,residual,reason=:breakdown)
    end
    (;x,converged=false,iterations,residual,reason=:maxiter)
end

"""BiCGStab for real/complex nonsymmetric systems with right preconditioning.
Reports true residuals and explicit breakdown/maxiter termination.
"""
function bicgstab(A::AbstractMatrix{T},b::AbstractVector{T};x0=zeros(T,length(b)),maxiter=2length(b),
                  rtol=sqrt(eps(real(T))),atol=zero(real(T)),preconditioner=nothing) where {T<:NativeFloat}
    target=_krylov_setup(A,b,x0,rtol,atol,maxiter)
    _check_preconditioner(preconditioner,length(b))
    x=copy(x0); r=similar(b); temp=similar(b); residual=_true_residual!(r,A,x,b,temp)
    residual<=target && return (;x,converged=true,iterations=0,residual,reason=:converged)
    shadow=copy(r); p=zeros(T,length(b)); v=similar(p); fill!(v,zero(T))
    s=similar(p); t=similar(p); phat=similar(p); shat=similar(p)
    oldrho=one(T); α=one(T); ω=one(T)
    for iteration in 1:maxiter
        ρ=dotc(shadow,r)
        if iszero(ρ) || !isfinite(ρ) || iszero(ω)
            residual=_true_residual!(r,A,x,b,temp)
            return (;x,converged=false,iterations=iteration-1,residual,reason=:breakdown)
        end
        β=(ρ/oldrho)*(α/ω)
        for i in eachindex(p)
            p[i]=r[i]+β*(p[i]-ω*v[i])
        end
        _precondition!(phat,p,preconditioner); _native_mv!(v,A,phat)
        den=dotc(shadow,v)
        if iszero(den) || !isfinite(den)
            residual=_true_residual!(r,A,x,b,temp)
            return (;x,converged=false,iterations=iteration-1,residual,reason=:breakdown)
        end
        α=ρ/den
        for i in eachindex(s)
            s[i]=r[i]-α*v[i]
        end
        if nrm2(s)<=target
            axpy!(x,phat,α); residual=_true_residual!(r,A,x,b,temp)
            residual<=target && return (;x,converged=true,iterations=iteration,residual,reason=:converged)
            # Residual replacement restarts the recurrence if roundoff misled it.
            copyto!(shadow,r); fill!(p,zero(T)); fill!(v,zero(T)); oldrho=α=ω=one(T)
            continue
        end
        _precondition!(shat,s,preconditioner); _native_mv!(t,A,shat)
        tt=real(dotc(t,t))
        if iszero(tt) || !isfinite(tt)
            residual=_true_residual!(r,A,x,b,temp)
            return (;x,converged=false,iterations=iteration-1,residual,reason=:breakdown)
        end
        ω=dotc(t,s)/tt
        for i in eachindex(x)
            x[i]+=α*phat[i]+ω*shat[i]
        end
        residual=_true_residual!(r,A,x,b,temp)
        residual<=target && return (;x,converged=true,iterations=iteration,residual,reason=:converged)
        oldrho=ρ
    end
    residual=_true_residual!(r,A,x,b,temp)
    (;x,converged=false,iterations=maxiter,residual,reason=:maxiter)
end

"""Fully reorthogonalized Hermitian Lanczos projection and Ritz eigenpairs.
Returns explicit residuals and convergence flags. `which` is :smallest or
:largest algebraic. A supplied start vector determines the explored subspace.
"""
function lanczos(A::AbstractMatrix{T};nev=1,maxiter=min(size(A,1),max(20,4nev)),which=:largest,
                 start=T[sin(i)+cos(3i) for i in 1:size(A,1)],rtol=sqrt(eps(real(T)))) where {T<:NativeFloat}
    n=_square(A); Base.require_one_based_indexing(start)
    0<nev<=maxiter<=n && length(start)==n && rtol>=0 || throw(ArgumentError("invalid Lanczos options"))
    which in (:smallest,:largest) || throw(ArgumentError("unknown Ritz ordering"))
    norm=nrm2(start); isfinite(norm) && norm>0 || throw(ArgumentError("nonzero finite start vector required"))
    V=zeros(T,n,maxiter); w=zeros(T,n); d=real(T)[]; e=real(T)[]; beta=zero(real(T)); iterations=0
    for i in 1:n
        V[i,1]=start[i]/norm
    end
    for j in 1:maxiter
        _native_mv!(w,A,view(V,:,j)); a=real(dotc(view(V,:,j),w)); push!(d,a)
        axpy!(w,view(V,:,j),-a)
        j>1 && axpy!(w,view(V,:,j-1),-beta)
        for _ in 1:2, k in 1:j
            h=dotc(view(V,:,k),w); axpy!(w,view(V,:,k),-h)
        end
        beta=nrm2(w); iterations=j
        (iszero(beta) || j==maxiter) && break
        push!(e,beta)
        for i in 1:n
            V[i,j+1]=w[i]/beta
        end
    end
    E=eigen_tridiagonal(d,e); order=sortperm(E.values;rev=which===:largest)
    selected=order[1:min(nev,length(order))]; values=E.values[selected]
    vectors=Matrix{T}(undef,n,length(selected)); gemm!(vectors,view(V,:,1:iterations),T.(E.vectors[:,selected]))
    residuals=real(T)[]
    for j in axes(vectors,2)
        _native_mv!(w,A,view(vectors,:,j)); axpy!(w,view(vectors,:,j),-values[j]); push!(residuals,nrm2(w))
    end
    flags=[residuals[i]<=rtol*max(one(real(T)),abs(values[i])) for i in eachindex(values)]
    (;values,vectors,residuals,converged=length(selected)==nev && all(flags),iterations)
end
