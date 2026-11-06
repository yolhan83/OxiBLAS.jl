# SPDX-License-Identifier: Apache-2.0
# SparseArrays interoperability. Every method here involves an OxiBLAS type
# (or extends an OxiBLAS function), so SparseArrays itself is not extended.
#
# SparseMatrixCSC is compressed-column storage: CSCMatrix(S) shares its three
# vectors (no copy, O(1)), and so does CSRMatrix(transpose(S)), the CSR form
# of the transpose. Other targets need a genuine reordering or format change
# and allocate, but never pass through a dense matrix. Explicit stored zeros
# are kept by the compressed formats; COOMatrix is canonical (duplicates
# summed, zeros removed) by its own definition.

const _OxiSparse=Union{CSRMatrix,CSCMatrix,COOMatrix,ExtendedSparse}
const _SparseCSCOp=Union{Transpose{<:Any,<:SparseMatrixCSC},Adjoint{<:Any,<:SparseMatrixCSC}}

# ---- SparseArrays -> OxiBLAS -------------------------------------------------

"""    CSCMatrix(S::SparseMatrixCSC)

Wrap `S` without copying: the result shares `S`'s column pointers, row
indices and values (mutating one is visible in the other). Index and value
types are preserved."""
CSCMatrix(S::SparseMatrixCSC{T,I}) where {T<:GenericFloat,I}=
    CSCMatrix(_Shared(),size(S,1),size(S,2),SparseArrays.getcolptr(S),SparseArrays.rowvals(S),SparseArrays.nonzeros(S))
# The stored indices are read from the field, as nonzeroinds itself does:
# nonzeroinds is owned by LinearAlgebra from Julia 1.14 (SparseArrays
# re-imports it) and is absent from LinearAlgebra before, so no qualified name
# is correct on every supported version.
function CSCMatrix(v::SparseVector{T,I}) where {T<:GenericFloat,I}
    CSCMatrix(_Shared(),length(v),1,I[1,SparseArrays.nnz(v)+1],getfield(v,:nzind),SparseArrays.nonzeros(v))
end
# The CSR form of transpose(S) is S's compressed-column storage: shared.
CSRMatrix(S::Transpose{T,<:SparseMatrixCSC{T}}) where {T<:GenericFloat}=
    CSRMatrix(_Shared(),size(S,1),size(S,2),SparseArrays.getcolptr(parent(S)),SparseArrays.rowvals(parent(S)),SparseArrays.nonzeros(parent(S)))
# The adjoint shares the pattern; its values are conjugated copies.
function CSRMatrix(S::Adjoint{T,<:SparseMatrixCSC{T}}) where {T<:GenericFloat}
    P=parent(S); values=SparseArrays.nonzeros(P)
    CSRMatrix(_Shared(),size(S,1),size(S,2),SparseArrays.getcolptr(P),SparseArrays.rowvals(P),T<:Real ? values : conj.(values))
end
function CSRMatrix(v::Transpose{T,<:SparseVector{T,I}}) where {T<:GenericFloat,I}
    p=parent(v); CSRMatrix(_Shared(),1,length(p),I[1,SparseArrays.nnz(p)+1],getfield(p,:nzind),SparseArrays.nonzeros(p))
end
function CSRMatrix(v::Adjoint{T,<:SparseVector{T,I}}) where {T<:GenericFloat,I}
    p=parent(v); values=SparseArrays.nonzeros(p)
    CSRMatrix(_Shared(),1,length(p),I[1,SparseArrays.nnz(p)+1],getfield(p,:nzind),T<:Real ? values : conj.(values))
end
CSRMatrix(S::SparseMatrixCSC{<:GenericFloat})=CSRMatrix(CSCMatrix(S))
CSRMatrix(v::SparseVector{<:GenericFloat})=CSRMatrix(CSCMatrix(v))
CSCMatrix(S::Union{Transpose{T,<:SparseMatrixCSC{T}},Adjoint{T,<:SparseMatrixCSC{T}}}) where {T<:GenericFloat}=CSCMatrix(CSRMatrix(S))
CSCMatrix(v::Union{Transpose{<:Any,<:SparseVector},Adjoint{<:Any,<:SparseVector}})=CSCMatrix(CSRMatrix(v))
COOMatrix(S::Union{SparseMatrixCSC,SparseVector,_SparseCSCOp})=COOMatrix(CSCMatrix(S))
for Format in (:DIAMatrix,:ELLMatrix,:BSRMatrix,:BSCMatrix,:HYBMatrix,:SELLMatrix)
    @eval $Format(S::Union{SparseMatrixCSC,_SparseCSCOp};kwargs...)=$Format(CSRMatrix(S);kwargs...)
end
# Value/index type conversions go through SparseArrays' own typed copy. The
# CSR form is the CSC storage of the (typed) transpose, so the index type is
# kept (the CSC-to-CSR conversion builds Int indices).
for Format in (:CSCMatrix,:COOMatrix)
    @eval $Format{T,I}(S::SparseMatrixCSC) where {T,I}=$Format(SparseMatrixCSC{T,I}(S))
end
function CSRMatrix{T,I}(S::SparseMatrixCSC) where {T,I}
    St=copy(transpose(SparseMatrixCSC{T,I}(S)))
    CSRMatrix(_Shared(),size(S,1),size(S,2),SparseArrays.getcolptr(St),SparseArrays.rowvals(St),SparseArrays.nonzeros(St))
end
Base.convert(::Type{F},S::Union{SparseMatrixCSC,SparseVector,_SparseCSCOp}) where {F<:Union{CSRMatrix,CSCMatrix,COOMatrix}}=F(S)

# ---- OxiBLAS -> SparseArrays -------------------------------------------------

# SparseMatrixCSC requires sorted row indices within each column; CSC built
# from user arrays may not be sorted.
function _csc_sorted(colptr,rowind)
    @inbounds for j in 1:length(colptr)-1, p in colptr[j]+1:colptr[j+1]-1
        rowind[p-1]<rowind[p] || return false
    end
    true
end
"""    SparseMatrixCSC(A::CSCMatrix)

Shares `A`'s storage when its row indices are sorted within each column (as
produced by OxiBLAS conversions); otherwise sorts into new arrays."""
function SparseMatrixCSC(A::CSCMatrix{T,I}) where {T,I}
    _csc_sorted(A.colptr,A.rowind) && return SparseMatrixCSC{T,I}(A.m,A.n,A.colptr,A.rowind,A.values)
    colptr=copy(A.colptr); rows=similar(A.rowind); vals=similar(A.values)
    for j in 1:A.n
        r=A.colptr[j]:A.colptr[j+1]-1; order=sortperm(view(A.rowind,r))
        rows[r]=view(A.rowind,r)[order]; vals[r]=view(A.values,r)[order]
    end
    SparseMatrixCSC{T,I}(A.m,A.n,colptr,rows,vals)
end
SparseMatrixCSC(A::CSRMatrix)=SparseMatrixCSC(CSCMatrix(A))
SparseMatrixCSC(A::COOMatrix)=SparseArrays.sparse(A.rows,A.cols,A.values,A.m,A.n)
SparseMatrixCSC(A::ExtendedSparse)=SparseMatrixCSC(CSCMatrix(A))
# transpose(CSR) is the shared CSC view of the same storage.
SparseMatrixCSC(A::Transpose{T,<:CSRMatrix{T}}) where T=SparseMatrixCSC(CSCMatrix(_Shared(),size(A,1),size(A,2),parent(A).rowptr,parent(A).colind,parent(A).values))
SparseMatrixCSC(A::Adjoint{T,<:CSRMatrix{T}}) where T=SparseMatrixCSC(CSCMatrix(_Shared(),size(A,1),size(A,2),parent(A).rowptr,parent(A).colind,T<:Real ? parent(A).values : conj.(parent(A).values)))
SparseMatrixCSC(A::Union{Transpose{<:Any,<:_OxiSparse},Adjoint{<:Any,<:_OxiSparse}})=SparseMatrixCSC(_op_csr(A))
SparseMatrixCSC{T,I}(A::Union{_OxiSparse,Transpose{<:Any,<:_OxiSparse},Adjoint{<:Any,<:_OxiSparse}}) where {T,I}=SparseMatrixCSC{T,I}(SparseMatrixCSC(A))
SparseArrays.sparse(A::_OxiSparse)=SparseMatrixCSC(A)
SparseArrays.sparse(A::Union{Transpose{<:Any,<:_OxiSparse},Adjoint{<:Any,<:_OxiSparse}})=SparseMatrixCSC(A)
Base.convert(::Type{S},A::_OxiSparse) where {S<:SparseMatrixCSC}=S(A)
# Transposed/adjoint OxiBLAS operands as CSR (pattern transposition).
function _op_csr(A::Union{Transpose,Adjoint})
    C=CSCMatrix(parent(A)); vals=A isa Adjoint && !(eltype(A)<:Real) ? conj.(C.values) : C.values
    CSRMatrix(_Shared(),size(A,1),size(A,2),C.colptr,C.rowind,vals)
end
"""    SparseVector(A::Union{CSCMatrix,CSRMatrix,COOMatrix})

The single column (or single row) of a sparse matrix as a SparseVector. A
CSC column with sorted indices shares its storage."""
function SparseVector(A::CSCMatrix{T,I}) where {T,I}
    A.n==1 || throw(DimensionMismatch("SparseVector needs a single-column sparse matrix"))
    _csc_sorted(A.colptr,A.rowind) && return SparseVector{T,I}(A.m,A.rowind,A.values)
    order=sortperm(A.rowind); SparseVector{T,I}(A.m,A.rowind[order],A.values[order])
end
function SparseVector(A::Union{CSRMatrix,COOMatrix,ExtendedSparse})
    size(A,2)==1 && return SparseVector(CSCMatrix(A))
    size(A,1)==1 || throw(DimensionMismatch("SparseVector needs a single row or column"))
    C=CSRMatrix(A); order=sortperm(C.colind)
    SparseVector(C.n,C.colind[order],C.values[order])
end

# ---- OxiBLAS sparse routines on SparseArrays operands ------------------------

_native_mv!(y,A::SparseMatrixCSC,x)=sparse_mul!(y,CSCMatrix(A),x)
sparse_mul!(y::AbstractVector,A::SparseMatrixCSC,x::AbstractVector,args...;kwargs...)=sparse_mul!(y,CSCMatrix(A),x,args...;kwargs...)
sparse_mul!(Y::AbstractMatrix,A::SparseMatrixCSC,X::AbstractMatrix,args...;kwargs...)=sparse_mul!(Y,CSCMatrix(A),X,args...;kwargs...)
function sparse_matmul(A::SparseMatrixCSC{T},B::SparseMatrixCSC{T};kwargs...) where T
    size(A,2)==size(B,1) || throw(DimensionMismatch("sparse matrix product dimensions differ"))
    C=sparse_matmul(CSRMatrix(transpose(B)),CSRMatrix(transpose(A));kwargs...)
    SparseMatrixCSC(CSCMatrix(_Shared(),size(A,1),size(B,2),C.rowptr,C.colind,C.values))
end
sparse_matmul(A::SparseMatrixCSC,B::Union{_OxiSparse,SparseMatrixCSC};kwargs...)=sparse_matmul(CSRMatrix(A),CSRMatrix(B);kwargs...)
sparse_matmul(A::_OxiSparse,B::SparseMatrixCSC;kwargs...)=sparse_matmul(CSRMatrix(A),CSRMatrix(B);kwargs...)
ilu0(A::SparseMatrixCSC)=ilu0(CSCMatrix(A))
SparseArrays.nnz(A::Union{CSRMatrix,CSCMatrix,COOMatrix})=length(A.values)

# ---- CSCMatrix storage accessors and copies ------------------------------------
# The public SparseArrays accessors on our own CSC storage.
SparseArrays.rowvals(A::CSCMatrix)=A.rowind
SparseArrays.nonzeros(A::CSCMatrix)=A.values
SparseArrays.nzrange(A::CSCMatrix,j::Integer)=A.colptr[j]:A.colptr[j+1]-1
# Copies keep the CSCMatrix type.
Base.copy(A::CSCMatrix)=CSCMatrix(_Shared(),A.m,A.n,copy(A.colptr),copy(A.rowind),copy(A.values))
Base.similar(A::CSCMatrix,::Type{S}) where {S<:GenericFloat}=CSCMatrix(_Shared(),A.m,A.n,copy(A.colptr),copy(A.rowind),similar(A.values,S))
Base.similar(A::CSCMatrix{T}) where T=similar(A,T)
Base.zero(A::CSCMatrix{T,I}) where {T,I}=CSCMatrix(_Shared(),A.m,A.n,ones(I,A.n+1),I[],T[])
# zero keeps the CSR/COO format too (AbstractSparseArray's zero is a
# SparseMatrixCSC; the other formats keep that sparse result).
Base.zero(A::CSRMatrix{T,I}) where {T,I}=CSRMatrix(_Shared(),A.m,A.n,ones(I,A.m+1),I[],T[])
Base.zero(A::COOMatrix{T,I}) where {T,I}=COOMatrix(A.m,A.n,I[],I[],T[])
