# SPDX-License-Identifier: Apache-2.0
# SparseArrays matrices are accepted wherever an OxiBLAS sparse matrix is:
# CSCMatrix(S) wraps their storage without copying (sparse_arrays.jl).
const NativeSparse=Union{CSRMatrix,CSCMatrix,COOMatrix,ExtendedSparse,SparseMatrixCSC}

"""
    sparse_triangular_solve(A::NativeSparse, b::AbstractVecOrMat; uplo=:L, trans=:N, unit=false) -> AbstractVecOrMat

Native sparse triangular substitution, with transpose/adjoint and unit
options. The compressed storage of the operand is used as it is (CSR rows or
CSC columns, a SparseMatrixCSC shared without copy); other formats are
converted to CSR once. A zero pivot throws SingularException before the
result is returned; the right-hand side is never modified.
"""
function sparse_triangular_solve(A::NativeSparse,b::AbstractVecOrMat{T};uplo=:L,trans=:N,unit=false) where {T<:GenericFloat}
    Base.require_one_based_indexing(b)
    _sparse_triangular_solve!(copy(b),A;uplo,trans,unit)
end
# In place on X (owned by the caller). Storage selection, then one kernel.
function _sparse_triangular_solve!(X::AbstractVecOrMat{T},A::NativeSparse;uplo=:L,trans=:N,unit=false) where T
    n=_square(A); upper=_uplo(uplo); tr=_trans_kind(trans)
    size(X,1)==n && eltype(A)===T || throw(DimensionMismatch("sparse triangular dimensions/type differ"))
    S=A isa SparseMatrixCSC ? CSCMatrix(A) : A isa Union{CSRMatrix,CSCMatrix} ? A : CSRMatrix(A)
    rowwise=S isa CSRMatrix
    ptr,ind,vals=rowwise ? (S.rowptr,S.colind,S.values) : (S.colptr,S.rowind,S.values)
    # op(A) is lower when exactly one of (upper, transposed) holds; the outer
    # index of the storage is the equation index (dot form) for CSR without
    # transposition and CSC with it, the unknown index (axpy form) otherwise.
    forward=upper==(tr!=0)
    # CSC storage has sorted indices (CSCMatrix invariant, SparseMatrixCSC):
    # each column splits once around its diagonal and the loops need no test.
    # The four flags become type parameters: one branch-free loop per case.
    rowwise ? _trisolve_kernel!(X,ptr,ind,vals,n,Val(tr==0),Val(forward),Val(tr==2),Val(unit)) :
              _trisolve_sorted!(X,ptr,ind,vals,n,Val(tr!=0),Val(forward),Val(tr==2),Val(unit))
end
function _trisolve_sorted!(X,ptr,ind,vals,n,::Val{dotform},::Val{forward},::Val{conjugated},::Val{unit}) where {dotform,forward,conjugated,unit}
    T=eltype(X)
    @inbounds for c in axes(X,2)
        for step in 1:n
            k=forward ? step : n+1-step
            lo,hi=ptr[k],ptr[k+1]-1
            # p: first entry with index >= k. The diagonal sits at the top of a
            # stored-lower column and at the bottom of a stored-upper one: the
            # search starts on that side.
            if dotform!=forward
                p=lo
                while p<=hi && Int(ind[p])<k
                    p+=1
                end
            else
                p=hi+1
                while p>lo && Int(ind[p-1])>=k
                    p-=1
                end
            end
            hasdiag=p<=hi && Int(ind[p])==k
            d=hasdiag ? (conjugated ? conj(vals[p]) : vals[p]) : zero(T)
            !unit && iszero(d) && throw(SingularException(k))
            # Entries strictly before / after the diagonal.
            before=lo:p-1; after=(hasdiag ? p+1 : p):hi
            if dotform
                s=X[k,c]
                for q in (forward ? before : after)
                    v=conjugated ? conj(vals[q]) : vals[q]
                    s=muladd(-v,X[Int(ind[q]),c],s)
                end
                X[k,c]=unit ? s : s/d
            else
                xk=unit ? X[k,c] : X[k,c]/d
                X[k,c]=xk
                iszero(xk) && continue
                for q in (forward ? after : before)
                    v=conjugated ? conj(vals[q]) : vals[q]
                    m=Int(ind[q]); X[m,c]=muladd(-v,xk,X[m,c])
                end
            end
        end
    end
    X
end
# Dot form: x[k] = (b[k] - Σ v·x[m]) / d_k over the stored m on the solved
# side of k. Axpy form: x[k] /= d_k, then x[m] -= v·x[k]. Entries on the other
# side of the diagonal are ignored, as is the diagonal when `unit`.
function _trisolve_kernel!(X,ptr,ind,vals,n,::Val{dotform},::Val{forward},::Val{conjugated},::Val{unit}) where {dotform,forward,conjugated,unit}
    T=eltype(X)
    @inbounds for c in axes(X,2)
        for step in 1:n
            k=forward ? step : n+1-step
            lo,hi=ptr[k],ptr[k+1]-1
            if dotform
                s=X[k,c]; d=zero(T)      # a missing diagonal is a zero pivot
                for p in lo:hi
                    m=Int(ind[p]); v=conjugated ? conj(vals[p]) : vals[p]
                    if m==k
                        d=v
                    elseif forward ? m<k : m>k
                        s=muladd(-v,X[m,c],s)
                    end
                end
                if !unit
                    iszero(d) && throw(SingularException(k)); s/=d
                end
                X[k,c]=s
            else
                xk=X[k,c]
                if !unit
                    d=zero(T)
                    for p in lo:hi
                        Int(ind[p])==k && (d=conjugated ? conj(vals[p]) : vals[p]; break)
                    end
                    iszero(d) && throw(SingularException(k)); xk/=d; X[k,c]=xk
                end
                iszero(xk) && continue
                for p in lo:hi
                    m=Int(ind[p])
                    if forward ? m>k : m<k
                        v=conjugated ? conj(vals[p]) : vals[p]
                        X[m,c]=muladd(-v,xk,X[m,c])
                    end
                end
            end
        end
    end
    X
end

struct SparseLUFactor{T,R}
    L::CSRMatrix{T,Int}
    U::CSRMatrix{T,Int}
    p::Vector{Int}
    q::Vector{Int}
    drop_tolerance::R
end
# Natural column order (q = 1:n).
SparseLUFactor(L::CSRMatrix{T,Int},U::CSRMatrix{T,Int},p::Vector{Int},drop_tolerance::R) where {T,R}=
    SparseLUFactor{T,R}(L,U,p,collect(1:length(p)),drop_tolerance)
# Real scalar parameters are carried separately to keep field types concrete.
function _rows_to_csr(rows,::Type{T},n) where T
    m=length(rows)
    ptr=Vector{Int}(undef,m+1); ptr[1]=1
    cols=Int[]; values=T[]; rowcols=Int[]
    for (i,row) in enumerate(rows)
        empty!(rowcols)
        for (j,value) in row
            !iszero(value) && push!(rowcols,j)
        end
        sort!(rowcols)
        for j in rowcols
            push!(cols,j); push!(values,row[j])
        end
        ptr[i+1]=length(values)+1
    end
    CSRMatrix(m,n,ptr,cols,values)
end
"""
    sparse_lu(A::NativeSparse; droptol=zero(real(eltype(A))), maxfill=typemax(Int), pivot_threshold=one(real(eltype(A))), ordering=:auto) -> SparseLUFactor

Sparse LU with threshold partial row pivoting, A[p,q] = L·U.

`ordering` (exact factor):
- `:auto` (default): as `multifrontal_lu` (UMFPACK's symmetric strategy for a
  mostly symmetric pattern, singletons and AMD(AᵀA) otherwise);
- `:natural`: q = 1:n, left-looking Gilbert–Peierls;
- `:colamd`, `:amd_ata` or a permutation vector: Gilbert–Peierls in that
  column order.

`droptol>0` gives threshold incomplete LU in the natural order, with
`maxfill` bounding the retained lower/upper entries per row, by dictionary
rows. No dense expansion.
"""
function sparse_lu(A::NativeSparse;droptol=zero(real(eltype(A))),maxfill=typemax(Int),pivot_threshold=one(real(eltype(A))),ordering=:auto)
    n=_square(A); T=eltype(A); R=real(T)
    droptol>=0 && maxfill>=0 && 0<=pivot_threshold<=1 || throw(ArgumentError("invalid sparse LU options"))
    if iszero(droptol)
        Cc=_as_csc(A); all(isfinite,Cc.values) || throw(ArgumentError("finite sparse LU entries required"))
        if ordering===:auto
            F=multifrontal_lu(Cc;pivot_threshold)
            return SparseLUFactor(F.L,F.U,F.p,F.q,R(droptol))
        end
        q=ordering===:natural ? nothing : ordering===:amd_ata ? amd_ordering(Cc;kind=:ata_sparse) :
          ordering===:colamd ? colamd_ordering(Cc) :
          ordering isa AbstractVector ? _checked_sparse_permutation(ordering,n) : throw(ArgumentError("unknown sparse LU ordering"))
        L,U,p=_gp_lu(Cc,R(pivot_threshold),q)
        return SparseLUFactor(L,U,p,q===nothing ? collect(1:n) : q,R(droptol))
    end
    ordering in (:auto,:natural) || throw(ArgumentError("incomplete LU uses the natural column order"))
    C=CSRMatrix(A)
    rows=[Dict{Int,T}() for _ in 1:n]; p=collect(1:n)
    for i in 1:n, k in C.rowptr[i]:C.rowptr[i+1]-1
        rows[i][C.colind[k]]=C.values[k]
    end
    for j in 1:n
        pivot=j; largest=abs(get(rows[j],j,zero(T)))
        for i in j+1:n
            value=abs(get(rows[i],j,zero(T)))
            value>largest && (largest=value; pivot=i)
        end
        iszero(largest) && throw(SingularException(j))
        current=abs(get(rows[j],j,zero(T)))
        # A zero threshold may retain any nonzero diagonal, never a zero one.
        !iszero(current) && current>=pivot_threshold*largest && (pivot=j)
        if pivot!=j
            rows[j],rows[pivot]=rows[pivot],rows[j]; p[j],p[pivot]=p[pivot],p[j]
        end
        diagonal=rows[j][j]
        for i in j+1:n
            value=get(rows[i],j,zero(T)); iszero(value) && continue
            l=value/diagonal; rows[i][j]=l
            scale=maximum(abs,values(rows[i]);init=zero(R))
            for (col,u) in rows[j]
                col>j || continue
                updated=get(rows[i],col,zero(T))-l*u
                if iszero(updated) || (droptol>0 && abs(updated)<=droptol*scale)
                    delete!(rows[i],col)
                else
                    rows[i][col]=updated
                end
            end
        end
    end
    lower=[Dict{Int,T}(i=>one(T)) for i in 1:n]; upper=[Dict{Int,T}() for _ in 1:n]
    for i in 1:n
        lowkeys=[j for j in keys(rows[i]) if j<i]; upkeys=[j for j in keys(rows[i]) if j>i]
        if droptol>0 && maxfill<typemax(Int)
            sort!(lowkeys;by=j->abs(rows[i][j]),rev=true); resize!(lowkeys,min(length(lowkeys),maxfill))
            sort!(upkeys;by=j->abs(rows[i][j]),rev=true); resize!(upkeys,min(length(upkeys),maxfill))
        end
        for j in lowkeys
            lower[i][j]=rows[i][j]
        end
        upper[i][i]=rows[i][i]
        for j in upkeys
            upper[i][j]=rows[i][j]
        end
    end
    SparseLUFactor(_rows_to_csr(lower,T,n),_rows_to_csr(upper,T,n),p,R(droptol))
end
ilut(A::NativeSparse;droptol=1e-3,maxfill=20,kwargs...)=sparse_lu(A;droptol,maxfill,kwargs...)
function solve(F::SparseLUFactor,b::AbstractVecOrMat)
    n=size(F.L,1); Base.require_one_based_indexing(b)
    size(b,1)==n || throw(DimensionMismatch("sparse LU solve dimensions differ"))
    X=ndims(b)==1 ? b[F.p] : b[F.p,:]     # the permuted copy is the workspace
    if iszero(F.drop_tolerance) && eltype(X)===eltype(F.L)
        # Exact factor: unit diagonal last in each L row, pivot first in each
        # U row (src/sparse/supernodal_lu.jl).
        for j in axes(X,2)
            _lu_csr_solve!(view(X,:,j),F.L,F.U)
        end
    else
        _sparse_triangular_solve!(X,F.L;uplo=:L,unit=true)
        _sparse_triangular_solve!(X,F.U;uplo=:U)
    end
    # x[q] = solution in the column order.
    F.q==1:n && return X
    Y=similar(X)
    ndims(b)==1 ? (Y[F.q]=X) : (Y[F.q,:]=X)
    Y
end
function _precondition!(z,r,F::SparseLUFactor)
    n=length(z); length(r)==n==length(F.p) || throw(DimensionMismatch("sparse LU preconditioner dimensions differ"))
    rr=Base.mightalias(z,r) ? copy(r) : r; q=F.q
    # Solve in the column-permuted positions z[q[i]] directly so no temporary
    # copy(z) is allocated when q != 1:n and no mutable workspace is shared.
    @inbounds for i in 1:n
        z[q[i]]=rr[F.p[i]]
    end
    @inbounds for i in 1:n
        qi=q[i]; zi=z[qi]
        for p in F.L.rowptr[i]:F.L.rowptr[i+1]-1
            j=F.L.colind[p]; j<i && (zi-=F.L.values[p]*z[q[j]])
        end
        z[qi]=zi
    end
    @inbounds for i in n:-1:1
        qi=q[i]; zi=z[qi]; diagonal=zero(eltype(z))
        for p in F.U.rowptr[i]:F.U.rowptr[i+1]-1
            j=F.U.colind[p]
            j>i ? (zi-=F.U.values[p]*z[q[j]]) : j==i && (diagonal=F.U.values[p])
        end
        z[qi]=zi/diagonal
    end
    z
end
_check_preconditioner(F::SparseLUFactor,n)=size(F.L)==(n,n) ? nothing : throw(DimensionMismatch("preconditioner dimensions differ"))

struct SparseCholeskyFactor{T,R}
    L::CSCMatrix{T,Int}
    drop_tolerance::R
    incomplete::Bool
end
"""
    sparse_cholesky(A::NativeSparse; uplo=:L, fill=:full, droptol=zero(real(eltype(A)))) -> SparseCholeskyFactor

Sparse Cholesky in the natural order, A = L·Lᴴ. Fill is stored sparsely,
never in a dense matrix. The exact factor goes through the supernodal code of
`multifrontal_cholesky` (natural ordering; the postorder is undone);
`fill=:zero` gives IC(0) and positive droptol threshold IC, by left-looking
columns.
"""
function sparse_cholesky(A::NativeSparse;uplo=:L,fill=:full,droptol=zero(real(eltype(A))))
    n=_square(A); upper=_uplo(uplo); T=eltype(A); R=real(T)
    fill in (:full,:zero) && droptol>=0 || throw(ArgumentError("invalid sparse Cholesky options"))
    if fill===:full && iszero(droptol) && T<:NativeFloat
        F=multifrontal_cholesky(A;uplo,ordering=:natural)
        return SparseCholeskyFactor(_chol_unpermuted(F.L,F.p),R(droptol),false)
    end
    C=CSRMatrix(A)
    columns=[Dict{Int,T}() for _ in 1:n]
    for i in 1:n, p in C.rowptr[i]:C.rowptr[i+1]-1
        j=C.colind[p]; value=C.values[p]
        if upper ? i<=j : i>=j
            row,col=upper ? (j,i) : (i,j)
            columns[col][row]=row==col ? T(real(value)) : (upper ? conj(value) : value)
        end
    end
    lowerrows=[Dict{Int,T}() for _ in 1:n]; colrows=Vector{Vector{Int}}(); colvalues=Vector{Vector{T}}()
    for j in 1:n
        w=copy(columns[j]); pattern=Set(keys(w))
        for (k,ljk) in lowerrows[j]
            k<j || continue
            for (i,lik) in zip(colrows[k],colvalues[k])
                i>=j || continue
                fill===:zero && !(i in pattern) && continue
                w[i]=get(w,i,zero(T))-lik*conj(ljk)
            end
        end
        diagonal=real(get(w,j,zero(T)))
        isfinite(diagonal) && diagonal>0 || throw(PosDefException(j))
        root=sqrt(diagonal); rows=Int[j]; vals=T[root]; lowerrows[j][j]=T(root)
        scale=maximum(abs,Base.values(w);init=zero(R))
        for i in sort!(collect(keys(w)))
            i>j || continue
            value=w[i]/root
            if !iszero(value) && abs(w[i])>droptol*scale
                push!(rows,i); push!(vals,value); lowerrows[i][j]=value
            end
        end
        push!(colrows,rows); push!(colvalues,vals)
    end
    ptr=Int[1]; indices=Int[]; values=T[]
    for j in 1:n
        append!(indices,colrows[j]); append!(values,colvalues[j]); push!(ptr,length(values)+1)
    end
    SparseCholeskyFactor(CSCMatrix(n,n,ptr,indices,values),R(droptol),fill===:zero || droptol>0)
end
ic0(A::NativeSparse;kwargs...)=sparse_cholesky(A;fill=:zero,kwargs...)
ict(A::NativeSparse;droptol=1e-3,kwargs...)=sparse_cholesky(A;droptol,kwargs...)
function solve(F::SparseCholeskyFactor,b::AbstractVecOrMat)
    Base.require_one_based_indexing(b)
    X=copy(b)
    if !F.incomplete && eltype(X)===eltype(F.L) && size(X,1)==size(F.L,1)
        # Exact factor: by fundamental supernodes (an incomplete factor need
        # not have their nested patterns).
        nodes=_chol_fundamental_nodes(F.L); y=similar(X,size(X,1))
        for j in axes(X,2)
            _chol_solve!(view(X,:,j),y,F.L,nodes)
        end
        return X
    end
    _sparse_triangular_solve!(X,F.L;uplo=:L)
    _sparse_triangular_solve!(X,F.L;uplo=:L,trans=:C)
end
function _precondition!(z,r,F::SparseCholeskyFactor)
    length(z)==length(r)==size(F.L,1) || throw(DimensionMismatch("sparse Cholesky preconditioner dimensions differ"))
    copyto!(z,r); A=F.L; n=size(A,1)
    for j in 1:n
        z[j]/=A.values[A.colptr[j]]
        for p in A.colptr[j]+1:A.colptr[j+1]-1
            z[A.rowind[p]]-=A.values[p]*z[j]
        end
    end
    for j in n:-1:1
        for p in A.colptr[j]+1:A.colptr[j+1]-1
            z[j]-=conj(A.values[p])*z[A.rowind[p]]
        end
        z[j]/=conj(A.values[A.colptr[j]])
    end
    z
end
_check_preconditioner(F::SparseCholeskyFactor,n)=size(F.L)==(n,n) ? nothing : throw(DimensionMismatch("preconditioner dimensions differ"))
