# SPDX-License-Identifier: Apache-2.0
struct _FrontalUpdate{T}
    indices::Vector{Int}
    values::Matrix{T} # lower triangle only
end
struct MultifrontalCholeskyFactor{T}
    L::CSCMatrix{T,Int}
    p::Vector{Int}
    parents::Vector{Int}
    nodes::Vector{UnitRange{Int}}
    front_sizes::Vector{Int}
    work::Vector{T}
    guard::ReentrantLock
end
function _sparse_hermitian(A,uplo)
    C=CSRMatrix(A); upper=_uplo(uplo); T=eltype(A); rows=Int[]; cols=Int[]; values=T[]
    for i in 1:C.m, p in C.rowptr[i]:C.rowptr[i+1]-1
        j=C.colind[p]; (upper ? i<=j : i>=j) || continue
        value=i==j ? T(real(C.values[p])) : C.values[p]
        push!(rows,i); push!(cols,j); push!(values,value)
        if i!=j
            push!(rows,j); push!(cols,i); push!(values,conj(value))
        end
    end
    CSRMatrix(COOMatrix(C.m,C.n,rows,cols,values))
end
function _supernodes(columns,maxwidth)
    nodes=UnitRange{Int}[]; n=length(columns); first=1
    while first<=n
        last=first
        while last<n && last-first+1<maxwidth && columns[last][2:end]==columns[last+1]
            last+=1
        end
        push!(nodes,first:last); first=last+1
    end
    nodes
end
# multifrontal_cholesky: src/sparse/supernodal_cholesky.jl
supernodal_cholesky(A;kwargs...)=multifrontal_cholesky(A;supernodal=true,kwargs...)
function _precondition!(z,r,F::MultifrontalCholeskyFactor)
    n=length(F.p); length(z)==length(r)==n || throw(DimensionMismatch("multifrontal solve dimensions differ"))
    lock(F.guard) do
        # work: permuted right-hand side, then the supernode buffer.
        x=view(F.work,1:n); y=view(F.work,n+1:2n)
        @inbounds for i in 1:n
            x[i]=r[F.p[i]]
        end
        _chol_solve!(x,y,F.L,F.nodes)
        @inbounds for i in 1:n
            z[F.p[i]]=x[i]
        end
        z
    end
end
function solve(F::MultifrontalCholeskyFactor{T},b::AbstractVecOrMat{T}) where T
    size(b,1)==length(F.p) || throw(DimensionMismatch("multifrontal solve dimensions differ"))
    if ndims(b)==1
        return _precondition!(similar(b),b,F)
    end
    X=similar(b)
    for j in axes(b,2)
        _precondition!(view(X,:,j),view(b,:,j),F)
    end
    X
end
_check_preconditioner(F::MultifrontalCholeskyFactor,n)=length(F.p)==n ? nothing : throw(DimensionMismatch("preconditioner dimensions differ"))

struct SparseBlockedLUFactor{T,R}
    factor::SparseLUFactor{T,R}
    panels::Vector{UnitRange{Int}}
end
"""Sparse blocked right-looking LU. Sparse pivot panels are factored in
column dictionaries; only the active L21/U12 update front is packed densely
for native TRSM/GEMM. Global A is never expanded to an n×n dense matrix.
"""
function sparse_lu_blocked(A::NativeSparse;blocksize=24,threaded=true)
    n=_square(A); blocksize>=1 || throw(ArgumentError("positive sparse LU block size required"))
    C=CSRMatrix(A); T=eltype(A); rows=[Dict{Int,T}() for _ in 1:n]; p=collect(1:n)
    for i in 1:n, k in C.rowptr[i]:C.rowptr[i+1]-1
        rows[i][C.colind[k]]=C.values[k]
    end
    panels=[j:min(n,j+blocksize-1) for j in 1:blocksize:n]
    for panel in panels
        firstcol,lastcol=first(panel),last(panel); q=length(panel)
        for j in panel
            pivot=j; largest=abs(get(rows[j],j,zero(T)))
            for i in j+1:n
                value=abs(get(rows[i],j,zero(T)))
                value>largest && (largest=value; pivot=i)
            end
            iszero(largest) && throw(SingularException(j))
            if pivot!=j
                rows[j],rows[pivot]=rows[pivot],rows[j]; p[j],p[pivot]=p[pivot],p[j]
            end
            diagonal=rows[j][j]
            for i in j+1:n
                value=get(rows[i],j,zero(T)); iszero(value) && continue
                l=value/diagonal; rows[i][j]=l
                for col in j+1:lastcol
                    update=get(rows[i],col,zero(T))-l*get(rows[j],col,zero(T))
                    iszero(update) ? delete!(rows[i],col) : (rows[i][col]=update)
                end
            end
        end
        lastcol==n && continue
        columns=sort!(unique([col for j in panel for col in keys(rows[j]) if col>lastcol])); isempty(columns) && continue
        L11=zeros(T,q,q); U12=Matrix{T}(undef,q,length(columns))
        for j in 1:q, i in j:q
            L11[i,j]=i==j ? one(T) : get(rows[firstcol+i-1],firstcol+j-1,zero(T))
        end
        for (j,col) in enumerate(columns), (i,row) in enumerate(panel)
            U12[i,j]=get(rows[row],col,zero(T))
        end
        trsm!(U12,L11;uplo=:L,unit=true,threaded)
        for (j,col) in enumerate(columns), (i,row) in enumerate(panel)
            value=U12[i,j]; iszero(value) ? delete!(rows[row],col) : (rows[row][col]=value)
        end
        active=[i for i in lastcol+1:n if any(!iszero(get(rows[i],col,zero(T))) for col in panel)]
        isempty(active) && continue
        L21=T[get(rows[i],j,zero(T)) for i in active,j in panel]
        update=T[get(rows[i],j,zero(T)) for i in active,j in columns]
        gemm!(update,L21,U12,-one(T),one(T);threaded)
        for (jj,col) in enumerate(columns), (ii,row) in enumerate(active)
            value=update[ii,jj]; iszero(value) ? delete!(rows[row],col) : (rows[row][col]=value)
        end
    end
    lower=[Dict{Int,T}(i=>one(T)) for i in 1:n]; upper=[Dict{Int,T}() for _ in 1:n]
    for i in 1:n, (j,value) in rows[i]
        j<i ? (lower[i][j]=value) : (upper[i][j]=value)
    end
    F=SparseLUFactor(_rows_to_csr(lower,T,n),_rows_to_csr(upper,T,n),p,zero(real(T)))
    SparseBlockedLUFactor(F,panels)
end
solve(F::SparseBlockedLUFactor,b::AbstractVecOrMat)=solve(F.factor,b)
_precondition!(z,r,F::SparseBlockedLUFactor)=_precondition!(z,r,F.factor)
_check_preconditioner(F::SparseBlockedLUFactor,n)=_check_preconditioner(F.factor,n)
