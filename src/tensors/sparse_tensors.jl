# SPDX-License-Identifier: Apache-2.0
# N-dimensional sparse tensors and the sparse paths of einsum/contract.

"""    SparseTensor{T,N} <: SparseArrays.AbstractSparseArray{T,Int,N}

Coordinate sparse tensor of any order: stored entries are kept as sorted,
unique column-major linear indices with their values. Explicit stored zeros
are preserved; `reshape` only changes the dimensions (the linear indices are
the same). Construct from coordinates (duplicates combined with `combine`),
from a dense array (`droptol`), or from SparseMatrixCSC, SparseVector and the
OxiBLAS sparse formats without any dense intermediate.
"""
struct SparseTensor{T,N} <: SparseArrays.AbstractSparseArray{T,Int,N}
    dims::NTuple{N,Int}
    indices::Vector{Int}
    values::Vector{T}
    function SparseTensor{T,N}(dims::NTuple{N,Integer},indices::Vector{Int},values::Vector{T}) where {T,N}
        all(>=(0),dims) || throw(ArgumentError("negative sparse tensor dimension"))
        length(indices)==length(values) || throw(DimensionMismatch("sparse tensor index/value lengths differ"))
        total=prod(BigInt.(dims);init=BigInt(1)); total<=typemax(Int) || throw(ArgumentError("sparse tensor too large for Int linear indices"))
        @inbounds for p in eachindex(indices)
            1<=indices[p]<=total || throw(ArgumentError("sparse tensor index out of range"))
            p>1 && indices[p-1]>=indices[p] && throw(ArgumentError("sparse tensor indices must be sorted and unique"))
        end
        new{T,N}(Int.(dims),indices,values)
    end
    # Internal: indices already sorted, unique and in range by construction
    # (the einsum kernels below); skips the O(nnz) validation pass.
    SparseTensor{T,N}(::_Shared,dims::NTuple{N,Int},indices::Vector{Int},values::Vector{T}) where {T,N}=new{T,N}(dims,indices,values)
end
SparseTensor(dims::NTuple{N,Integer},indices::Vector{Int},values::Vector{T}) where {T,N}=SparseTensor{T,N}(dims,indices,values)
Base.size(A::SparseTensor)=A.dims
Base.IndexStyle(::Type{<:SparseTensor})=IndexLinear()
Base.dataids(A::SparseTensor)=(Base.dataids(A.indices)...,Base.dataids(A.values)...)
function Base.getindex(A::SparseTensor{T},i::Int) where T
    @boundscheck checkbounds(A,i)
    p=searchsortedfirst(A.indices,i)
    p<=length(A.indices) && A.indices[p]==i ? A.values[p] : zero(T)
end
"""Assigning a value stores it (an explicit zero included); O(nnz) insertion."""
function Base.setindex!(A::SparseTensor,v,i::Int)
    @boundscheck checkbounds(A,i)
    p=searchsortedfirst(A.indices,i)
    if p<=length(A.indices) && A.indices[p]==i
        A.values[p]=v
    else
        insert!(A.indices,p,i); insert!(A.values,p,v)
    end
    A
end
Base.copy(A::SparseTensor)=SparseTensor(A.dims,copy(A.indices),copy(A.values))
Base.conj(A::SparseTensor{<:Real})=A
Base.conj(A::SparseTensor)=SparseTensor(A.dims,copy(A.indices),conj.(A.values))
Base.similar(A::SparseTensor{T},::Type{S},dims::Dims{N}) where {T,S,N}=SparseTensor(dims,Int[],S[])
Base.reshape(A::SparseTensor,dims::Dims)=(prod(dims;init=1)==length(A) || throw(DimensionMismatch("reshape changes the element count"));SparseTensor(dims,copy(A.indices),copy(A.values)))
function Base.Array(A::SparseTensor{T,N}) where {T,N}
    B=zeros(T,A.dims)
    @inbounds for p in eachindex(A.indices)
        B[A.indices[p]]=A.values[p]
    end
    B
end
SparseArrays.nnz(A::SparseTensor)=length(A.values)
SparseArrays.nonzeros(A::SparseTensor)=A.values
# Order 1 is an AbstractSparseVector (an alias of AbstractSparseArray{T,Ti,1}):
# its sorted linear indices are exactly the positions SparseArrays expects.
# nonzeroinds is owned by LinearAlgebra from Julia 1.14 (SparseArrays
# re-imports it) and by SparseArrays before: the method is added to the
# function object itself, whichever module owns it.
(::typeof(getfield(SparseArrays,:nonzeroinds)))(A::SparseTensor{<:Any,1})=A.indices
"""Stored coordinates (one tuple per entry, column-major order) and values."""
function SparseArrays.findnz(A::SparseTensor)
    C=CartesianIndices(A.dims)
    ([Tuple(C[i]) for i in A.indices],copy(A.values))
end
function SparseArrays.dropzeros(A::SparseTensor)
    keep=findall(!iszero,A.values); SparseTensor(A.dims,A.indices[keep],A.values[keep])
end
Base.:*(a::Number,A::SparseTensor)=SparseTensor(A.dims,copy(A.indices),a.*A.values)
Base.:*(A::SparseTensor,a::Number)=SparseTensor(A.dims,copy(A.indices),A.values.*a)
function Base.permutedims(A::SparseTensor{T,N},perm) where {T,N}
    p=Tuple(perm); isperm(p) && length(p)==N || throw(ArgumentError("invalid permutation"))
    dims=ntuple(d->A.dims[p[d]],N); C=CartesianIndices(A.dims); L=LinearIndices(dims)
    keys=[L[ntuple(d->C[i][p[d]],N)...] for i in A.indices]
    _sparse_tensor_from(dims,keys,copy(A.values))
end
Base.adjoint(A::SparseTensor{T,2}) where T=permutedims(conj(A),(2,1))
Base.transpose(A::SparseTensor{T,2}) where T=permutedims(A,(2,1))

# Unsorted, possibly repeated linear indices: sort and combine duplicates.
function _sparse_tensor_from(dims,keys::Vector{Int},values::Vector{T};combine=+) where T
    # Keys already in order (reductions that keep leading axes, full
    # reductions): combine adjacent duplicates, no sortperm.
    order=issorted(keys) ? eachindex(keys) : sortperm(keys); idx=Int[]; vals=T[]
    sizehint!(idx,length(keys)); sizehint!(vals,length(keys))
    for q in order
        k=keys[q]
        if !isempty(idx) && idx[end]==k
            vals[end]=combine(vals[end],values[q])
        else
            push!(idx,k); push!(vals,values[q])
        end
    end
    SparseTensor(dims,idx,vals)
end
"""    SparseTensor(dims, coordinates, values; combine=+)

`coordinates` is an N×nnz integer matrix or a vector of N-tuples /
CartesianIndex. Duplicate coordinates are combined with `combine`; stored
zeros (given or produced) are kept."""
function SparseTensor(dims::NTuple{N,Integer},coordinates,values::AbstractVector{T};combine=+) where {T,N}
    L=LinearIndices(Tuple(Int.(dims)))
    coords=coordinates isa AbstractMatrix ? (size(coordinates,1)==N || throw(DimensionMismatch("one coordinate row per dimension"));[Tuple(view(coordinates,:,p)) for p in axes(coordinates,2)]) : [Tuple(c) for c in coordinates]
    length(coords)==length(values) || throw(DimensionMismatch("coordinate and value counts differ"))
    keys=Vector{Int}(undef,length(coords))
    for (p,c) in enumerate(coords)
        length(c)==N || throw(DimensionMismatch("coordinate rank differs"))
        all(d->1<=c[d]<=dims[d],1:N) || throw(ArgumentError("sparse tensor coordinate out of range"))
        keys[p]=L[c...]
    end
    _sparse_tensor_from(Tuple(Int.(dims)),keys,collect(values);combine)
end
"""Stored entries of a dense array (|x| > droptol) as a SparseTensor."""
function SparseTensor(A::AbstractArray{T,N};droptol=zero(real(T))) where {T,N}
    Base.require_one_based_indexing(A); idx=Int[]; vals=T[]
    for (i,v) in enumerate(A)
        !(abs(v)<=droptol) && (push!(idx,i); push!(vals,v))
    end
    SparseTensor(size(A),idx,vals)
end
SparseTensor(A::SparseTensor)=copy(A)

# ---- Conversions between SparseTensor and matrix/vector sparse formats ------

# Rows are sorted within each column for SparseMatrixCSC (a SparseArrays
# invariant) and CSCMatrix (sorted by its constructor), so the column-major
# keys come out sorted and unique: no per-column check, no validation pass.
# The values are copied (the tensor never shares the matrix's storage).
function _sparse_tensor_csc(m,n,colptr,rowind,values::AbstractVector{T}) where T
    nz=Int(colptr[n+1])-1
    idx=Vector{Int}(undef,nz); vals=Vector{T}(undef,nz)
    @inbounds for j in 1:n
        base=(j-1)*m
        for p in colptr[j]:colptr[j+1]-1
            idx[p]=base+Int(rowind[p]); vals[p]=values[p]
        end
    end
    SparseTensor{T,2}(_Shared(),(Int(m),Int(n)),idx,vals)
end
SparseTensor(S::SparseMatrixCSC)=_sparse_tensor_csc(size(S,1),size(S,2),SparseArrays.getcolptr(S),SparseArrays.rowvals(S),SparseArrays.nonzeros(S))
SparseTensor(v::SparseVector)=SparseTensor((length(v),),Int.(getfield(v,:nzind)),copy(SparseArrays.nonzeros(v)))
SparseTensor(A::CSCMatrix)=_sparse_tensor_csc(A.m,A.n,A.colptr,A.rowind,A.values)
SparseTensor(A::Union{CSRMatrix,COOMatrix,ExtendedSparse})=SparseTensor(CSCMatrix(A))
SparseTensor(A::Union{Transpose{<:Any,<:Union{SparseMatrixCSC,_OxiSparse}},Adjoint{<:Any,<:Union{SparseMatrixCSC,_OxiSparse}}})=SparseTensor(CSCMatrix(_op_csr_any(A)))
SparseTensor(v::Union{Transpose{<:Any,<:SparseVector},Adjoint{<:Any,<:SparseVector}})=SparseTensor(CSCMatrix(CSRMatrix(v)))
_op_csr_any(A::Union{Transpose{<:Any,<:SparseMatrixCSC},Adjoint{<:Any,<:SparseMatrixCSC}})=CSRMatrix(A)
_op_csr_any(A)=_op_csr(A)

function CSCMatrix(A::SparseTensor{T,2}) where T
    m,n=A.dims; colptr=ones(Int,n+1); rows=Vector{Int}(undef,length(A.indices))
    for (p,i) in enumerate(A.indices)
        j=(i-1)÷max(m,1)+1; rows[p]=i-(j-1)*m; colptr[j+1]+=1
    end
    for j in 1:n
        colptr[j+1]+=colptr[j]-1
    end
    CSCMatrix(_Shared(),m,n,colptr,rows,copy(A.values))
end
CSRMatrix(A::SparseTensor{T,2}) where T=CSRMatrix(CSCMatrix(A))
COOMatrix(A::SparseTensor{T,2}) where T=COOMatrix(CSCMatrix(A))
SparseMatrixCSC(A::SparseTensor{T,2}) where T=SparseMatrixCSC(CSCMatrix(A))
SparseMatrixCSC{Tv,Ti}(A::SparseTensor{T,2}) where {T,Tv,Ti}=SparseMatrixCSC{Tv,Ti}(SparseMatrixCSC(A))
SparseVector(A::SparseTensor{T,1}) where T=SparseVector(A.dims[1],copy(A.indices),copy(A.values))
SparseArrays.sparse(A::SparseTensor{T,2}) where T=SparseMatrixCSC(A)
SparseArrays.sparse(A::SparseTensor{T,1}) where T=SparseVector(A)
Base.convert(::Type{SparseTensor},A::AbstractArray)=SparseTensor(A)
Base.convert(::Type{SparseTensor},A::SparseTensor)=A

# ---- Sparse operands of contractions ---------------------------------------

# Arrays treated as sparse by einsum/contract: converted to SparseTensor in
# O(nnz), never densified.
const SparseOperand=Union{SparseTensor,SparseMatrixCSC,SparseVector,_OxiSparse,
    Transpose{<:Any,<:Union{SparseMatrixCSC,SparseVector,_OxiSparse}},Adjoint{<:Any,<:Union{SparseMatrixCSC,SparseVector,_OxiSparse}}}
_is_sparse_operand(A)=A isa SparseOperand
_sparse_operand(A::SparseTensor)=A
_operand_nnz(A::SparseTensor)=length(A.indices)
_operand_nnz(A::Union{SparseMatrixCSC,SparseVector})=SparseArrays.nnz(A)
_operand_nnz(A::Union{CSRMatrix,CSCMatrix,COOMatrix})=length(A.values)
_operand_nnz(A::Union{Transpose,Adjoint})=_operand_nnz(parent(A))
_operand_nnz(A)=SparseArrays.nnz(_sparse_operand(A))
_sparse_operand(A::SparseOperand)=SparseTensor(A)
_sparse_operand(A)=nothing

# Coordinates of entry p along axis d (column-major), without CartesianIndex
# allocation in the inner loops.
# Column-major decoding by tail recursion over the dimensions: the former
# ntuple closure reassigned the captured remainder (a Core.Box per entry).
@inline _sparse_coords(i,dims::NTuple{N,Int}) where N=_sparse_coords_rest(i-1,dims)
@inline _sparse_coords_rest(r,::Tuple{})=()
@inline _sparse_coords_rest(r,dims::Tuple)=(q=r%first(dims); (q+1,_sparse_coords_rest(r÷first(dims),Base.tail(dims))...))

# Broadcast (size-1 axes of a label with a larger extent) and diagonals /
# exclusive reductions: the operand is brought to unique labels it shares
# with `keep` (the labels still needed: output or other operands).
function _sparse_unary(A::SparseTensor{T},labels,keep,dims) where T
    N=ndims(A)
    out=Int[]
    for l in labels
        (l in keep) && !(l in out) && push!(out,l)
    end
    # Axes whose extent is 1 while the label is larger broadcast: their
    # coordinate is constant and the label is not carried by this operand.
    carry=[d for d in 1:N if !(size(A,d)==1 && dims[labels[d]]!=1)]
    # Nothing to do (unique labels, all kept, no broadcast axis): the operand
    # as is. This is the usual case of every pairwise step.
    out==labels && length(carry)==N && return A,out
    axisof=Dict(l=>findfirst(d->labels[d]==l && d in carry,1:N) for l in out)
    any(l->axisof[l]===nothing,out) && (out=filter(l->axisof[l]!==nothing,out))
    outdims=Tuple(dims[l] for l in out)
    # Per axis: output multiplier (first carried axis of an output label), and
    # the axis it must equal (repeated labels: diagonals). Arithmetic decoding,
    # no tuples per entry.
    outmul=zeros(Int,N); same=collect(1:N); m=1
    for l in out
        d=axisof[l]::Int; outmul[d]=m; m*=dims[l]
    end
    for d in carry, e in carry
        e<d && labels[e]==labels[d] && same[d]==d && (same[d]=e)
    end
    coords=zeros(Int,N); keys=Int[]; vals=T[]; sizehint!(keys,length(A.indices)); sizehint!(vals,length(A.indices))
    sz=A.dims
    @inbounds for p in eachindex(A.indices)
        r=A.indices[p]-1
        for d in 1:N
            s=sz[d]; coords[d]=r%s; r=r÷s
        end
        ok=true
        for d in carry
            e=same[d]; e!=d && coords[d]!=coords[e] && (ok=false; break)
        end
        ok || continue
        key=1
        for d in 1:N; key+=coords[d]*outmul[d]; end
        push!(keys,key); push!(vals,A.values[p])
    end
    _sparse_tensor_from(outdims,keys,vals),out
end
# Dense counterpart: the existing scalar einsum path on one operand.
function _dense_unary(A,labels,keep,dims;threaded=true)
    out=Int[]
    for (d,l) in enumerate(labels)
        (l in keep) && !(l in out) && !(size(A,d)==1 && dims[l]!=1) && push!(out,l)
    end
    (labels==out && allunique(labels) && all(size(A,d)==dims[labels[d]] for d in eachindex(labels))) && return A,out
    _einsum_direct((A,),(labels,),out,dims,eltype(A);threaded),out
end

# Group codes of a SparseTensor's entries: each axis belongs to group 1, 2
# or 3, with a multiplier giving the entry's 0-based linear index inside its
# group. Decoding costs N divisions per entry, with no tuples or Dict.
function _group_codes(A::SparseTensor,labels,groups::NTuple{3,Vector{Int}},dims)
    N=ndims(A); grp=zeros(Int,N); mul=zeros(Int,N)
    for g in 1:3
        m=1
        for l in groups[g]
            d=findfirst(==(l),labels)::Int; grp[d]=g; mul[d]=m; m*=dims[l]
        end
    end
    n=length(A.indices); c1=Vector{Int}(undef,n); c2=Vector{Int}(undef,n); c3=Vector{Int}(undef,n)
    sz=A.dims
    @inbounds for p in 1:n
        r=A.indices[p]-1; a1=0; a2=0; a3=0
        for d in 1:N
            s=sz[d]; q=r%s; r=r÷s; g=grp[d]
            g==1 ? (a1+=q*mul[d]) : g==2 ? (a2+=q*mul[d]) : (a3+=q*mul[d])
        end
        c1[p]=a1; c2[p]=a2; c3[p]=a3
    end
    c1,c2,c3
end

"""Sparse × sparse: a column-wise SpGEMM (Gustavson) on the two operands
unfolded as matrices, per batch index. A is (free × contracted), B is
(contracted × free), the output column (r, b) is accumulated from the columns
(k, b) of A met in the column (r, b) of B. Computed zeros are kept."""
function _sparse_pair(A::SparseTensor{T},B::SparseTensor{T},la,lb,out,dims) where T
    batch=[l for l in out if l in la && l in lb]
    K=[l for l in la if l in lb && !(l in out)]
    L=[l for l in out if l in la && !(l in lb)]; R=[l for l in out if l in lb && !(l in la)]
    # Grouped output: [L R batch] directly, [R L batch] by exchanging roles;
    # anything else is computed in [L R batch] and then remapped.
    if out==vcat(R,L,batch) && out!=vcat(L,R,batch)
        return _sparse_pair_grouped(B,A,lb,la,R,K,L,batch,dims,out)
    end
    C=_sparse_pair_grouped(A,B,la,lb,L,K,R,batch,dims,vcat(L,R,batch))
    vcat(L,R,batch)==out && return C
    perm=Tuple(findfirst(==(l),vcat(L,R,batch))::Int for l in out)
    permutedims(C,perm)
end
# Stable sort permutation of keys in 0:nkeys-1, with the start of each key
# (a column pointer): counting sort when the key space is comparable to the
# number of entries, otherwise sortperm (no pointer: binary search).
function _key_perm(keys::Vector{Int},nkeys::Int)
    if nkeys<=max(4*length(keys),1<<16)
        ptr=zeros(Int,nkeys+1)
        @inbounds for k in keys; ptr[k+2>nkeys+1 ? nkeys+1 : k+2]+=1; end
        ptr[1]=1
        @inbounds for c in 1:nkeys; ptr[c+1]+=ptr[c]; end
        perm=Vector{Int}(undef,length(keys)); next=copy(ptr)
        @inbounds for q in eachindex(keys)
            k=keys[q]; perm[next[k+1]]=q; next[k+1]+=1
        end
        return perm,ptr
    end
    sortperm(keys),Int[]
end
function _sparse_pair_grouped(A::SparseTensor{T},B::SparseTensor{T},la,lb,L,K,R,batch,dims,natural) where T
    p(G)=prod(dims[l] for l in G;init=1)
    m,k,n,nb=p(L),p(K),p(R),p(batch)
    al,ak,ab=_group_codes(A,la,(L,K,batch),dims)
    bk,br,bb=_group_codes(B,lb,(K,R,batch),dims)
    outdims=Tuple(dims[l] for l in natural)
    (isempty(al) || isempty(bk)) && return SparseTensor(outdims,Int[],T[])
    # A by column (b, k), B by column (b, r): stable counting sorts that also
    # give A's column pointer; huge key spaces fall back to sortperm, with
    # binary search over the non-empty columns.
    akey=Vector{Int}(undef,length(ak)); @inbounds for q in eachindex(ak); akey[q]=ak[q]+k*ab[q]; end
    bkey=Vector{Int}(undef,length(br)); @inbounds for q in eachindex(br); bkey[q]=br[q]+n*bb[q]; end
    pa,aptr=_key_perm(akey,k*nb); pb,_=_key_perm(bkey,n*nb)
    dense=!isempty(aptr)
    asorted=dense ? Int[] : akey[pa]
    # Row accumulator: dense when m is reasonable, else sort-and-merge.
    densem=m<=max(4*(length(akey)+length(bkey)),1<<20)
    sums=densem ? zeros(T,m) : T[]; marks=densem ? zeros(Int,m) : Int[]
    touched=Int[]; tvals=T[]; keys=Int[]; vals=T[]
    Avals=A.values; Bvals=B.values; mark=0
    q=1; nB=length(pb)
    @inbounds while q<=nB
        col=bkey[pb[q]]; mark+=1; empty!(touched); empty!(tvals)
        r=col%n; b=col÷n
        while q<=nB && bkey[pb[q]]==col
            e=pb[q]; vb=Bvals[e]; ckey=bk[e]+k*b
            lo,hi=if dense
                aptr[ckey+1],aptr[ckey+2]-1
            else
                f=searchsortedfirst(asorted,ckey); g=searchsortedlast(asorted,ckey); (f,g)
            end
            for t in lo:hi
                ea=pa[t]; row=al[ea]; v=Avals[ea]*vb
                if densem
                    if marks[row+1]!=mark
                        marks[row+1]=mark; sums[row+1]=v; push!(touched,row)
                    else
                        sums[row+1]+=v
                    end
                else
                    push!(touched,row); push!(tvals,v)
                end
            end
            q+=1
        end
        base=m*(r+n*b)
        if densem
            sort!(touched)
            for row in touched; push!(keys,base+row+1); push!(vals,sums[row+1]); end
        elseif !isempty(touched)
            ord=sortperm(touched); prev=-1
            for t in ord
                row=touched[t]
                if row==prev
                    vals[end]+=tvals[t]
                else
                    push!(keys,base+row+1); push!(vals,tvals[t]); prev=row
                end
            end
        end
    end
    SparseTensor{T,length(outdims)}(_Shared(),outdims,keys,vals)   # keys emitted sorted
end

"""Sparse × dense (either order): per batch index, the sparse operand as a CSR
matrix (free labels × contracted labels) times the dense operand reshaped to
(contracted × free), with the native CSR kernel. Dense result."""
function _sparse_dense_pair(A::SparseTensor{T},B::AbstractArray{T},la,lb,out,dims;threaded=true,allocator=nothing) where T
    batch=[l for l in la if l in lb && l in out]
    contracted=[l for l in la if l in lb && !(l in out)]
    left=[l for l in la if !(l in lb)]; right=[l for l in lb if !(l in la)]
    m=prod(dims[l] for l in left;init=1); k=prod(dims[l] for l in contracted;init=1)
    n=prod(dims[l] for l in right;init=1); nb=prod(dims[l] for l in batch;init=1)
    pb=[findfirst(==(l),lb)::Int for l in vcat(contracted,right,batch)]
    Bp=pb==collect(1:ndims(B)) ? B : permutedims(B,Tuple(pb))
    B3=reshape(Bp,k,n,nb)
    # No batch, and A's axes grouped as [free contracted] or [contracted free]:
    # A's linear index already is a CSC index of the unfolded matrix
    # (rows = first group, column = second), sorted column by column. One
    # division per entry, no decoding or sort, then the CSC kernel.
    if nb==1 && (la==vcat(left,contracted) || la==vcat(contracted,left))
        first_free=la==vcat(left,contracted)
        rows_dim,cols_dim=first_free ? (m,k) : (k,m)
        if cols_dim<=max(4*length(A.indices),1<<20)
            cp=zeros(Int,cols_dim+1); ri=Vector{Int}(undef,length(A.indices))
            @inbounds for (p,idx) in enumerate(A.indices)
                c,r=divrem(idx-1,rows_dim); ri[p]=r+1; cp[c+2]+=1
            end
            cp[1]=1; @inbounds for c in 1:cols_dim; cp[c+1]+=cp[c]; end
            Acsc=CSCMatrix(_Shared(),rows_dim,cols_dim,cp,ri,A.values)   # shares A.values (read only)
            C2=_csc_mul(Acsc,!first_free,view(B3,:,:,1);threaded)
            # Its own name: `natural` is also assigned below and captured by
            # the generators of both branches.
            nat2=vcat(left,right)
            reshaped=reshape(C2,Tuple(dims[l] for l in nat2))
            nat2==out && return reshaped
            return permutedims(reshaped,Tuple(findfirst(==(l),nat2)::Int for l in out))
        end
    end
    # CSR (free × contracted) per batch, built directly: group codes by
    # arithmetic, then one sort on (batch, row, column), which is the CSR order.
    # Entries are unique, so no duplicate (row, column) pairs.
    rl,ck,bt=_group_codes(A,la,(left,contracted,batch),dims)
    # Stable sort on (batch, row): entries arrive in linear (lexicographic)
    # order, and the contracted code grows with that order inside a row, so
    # each row's columns come out sorted (a valid CSR, no second sort).
    rkey=Vector{Int}(undef,length(rl)); @inbounds for q in eachindex(rl); rkey[q]=rl[q]+m*bt[q]; end
    ord,rptr=_key_perm(rkey,m*nb)
    if isempty(rptr)   # huge row space: sort on (batch, row, column) instead
        ord=sortperm([ck[q]+k*rkey[q] for q in eachindex(rkey)])
        rptr=zeros(Int,m*nb+1); for key in rkey; rptr[key+2]+=1; end
        rptr[1]=1; for c in 1:m*nb; rptr[c+1]+=rptr[c]; end
    end
    C=allocator===nothing ? Array{T}(undef,m,n,nb) : allocator((m,n,nb))
    Av=A.values
    for b in 0:nb-1
        lo=rptr[b*m+1]; hi=rptr[(b+1)*m+1]-1
        rowptr=Vector{Int}(undef,m+1)
        @inbounds for i in 1:m+1; rowptr[i]=rptr[b*m+i]-lo+1; end
        colind=Vector{Int}(undef,hi-lo+1); vals=Vector{T}(undef,hi-lo+1)
        @inbounds for t in lo:hi; e=ord[t]; colind[t-lo+1]=ck[e]+1; vals[t-lo+1]=Av[e]; end
        R=CSRMatrix(_Shared(),m,k,rowptr,colind,vals)   # valid by construction
        sparse_mul!(view(C,:,:,b+1),R,view(B3,:,:,b+1);threaded)
    end
    natural=vcat(left,right,batch)
    reshaped=reshape(C,Tuple(dims[l] for l in natural))
    natural==out && return reshaped
    perm=Tuple(findfirst(==(l),natural)::Int for l in out)
    allocator===nothing ? permutedims(reshaped,perm) : PermutedDimsArray(reshaped,perm)
end

# ---- 2-D sparse operands: thin layer over the matrix kernels -----------------
# A sparse matrix in CSC layout (SparseMatrixCSC, CSCMatrix) stays as it is in
# einsum: matrix-product steps and 2-D unary steps work on its colptr/rowval
# arrays directly. A CSRMatrix with labels (i,j) is the CSC of its transpose,
# so it enters with labels (j,i) at no cost. Anything else becomes a
# SparseTensor when a step needs it.
const _CSCLike=Union{SparseMatrixCSC,CSCMatrix}
_csc_parts(A::SparseMatrixCSC)=(size(A,1),size(A,2),SparseArrays.getcolptr(A),SparseArrays.rowvals(A),SparseArrays.nonzeros(A))
_csc_parts(A::CSCMatrix)=(A.m,A.n,A.colptr,A.rowind,A.values)
_as_sparse_tensor(A::_CSCLike)=SparseTensor(A)
_as_sparse_tensor(A)=A
# CSC transpose by counting (rows sorted within columns), O(nnz + m).
function _csc_transpose(A::_CSCLike)
    m,n,cp,ri,v=_csc_parts(A); T=eltype(v)
    ptr=zeros(Int,m+1)
    @inbounds for p in 1:cp[n+1]-1; ptr[ri[p]+1]+=1; end
    ptr[1]=1; @inbounds for i in 1:m; ptr[i+1]+=ptr[i]; end
    next=copy(ptr); rows=Vector{Int}(undef,cp[n+1]-1); vals=Vector{T}(undef,cp[n+1]-1)
    @inbounds for j in 1:n, p in cp[j]:cp[j+1]-1
        i=ri[p]; q=next[i]; rows[q]=j; vals[q]=v[p]; next[i]=q+1
    end
    CSCMatrix(_Shared(),n,m,ptr,rows,vals)
end
# C = A·B, column by column (Gustavson). Computed zeros are kept, as in the
# general einsum path; rows sorted within each column.
function _csc_spgemm(A::_CSCLike,B::_CSCLike)
    m,_,ap,ai,av=_csc_parts(A); _,n,bp,bi,bv=_csc_parts(B); T=promote_type(eltype(av),eltype(bv))
    colptr=Vector{Int}(undef,n+1); colptr[1]=1
    # Output buffers sized from the nonzero counts and grown by doubling;
    # the touched rows of a column live in a preallocated buffer (no push!).
    cap=max(Int(ap[end])+Int(bp[end]),16); rows=Vector{Int}(undef,cap); vals=Vector{T}(undef,cap)
    sums=zeros(T,m); marks=zeros(Int,m); touched=Vector{Int}(undef,m); nz=0
    @inbounds for j in 1:n
        t=0
        for q in bp[j]:bp[j+1]-1
            k=bi[q]; x=bv[q]
            for p in ap[k]:ap[k+1]-1
                i=ai[p]; y=av[p]*x
                if marks[i]!=j
                    marks[i]=j; sums[i]=y; t+=1; touched[t]=i
                else
                    sums[i]+=y
                end
            end
        end
        t>1 && sort!(view(touched,1:t))
        if nz+t>cap
            cap=max(2cap,nz+t); resize!(rows,cap); resize!(vals,cap)
        end
        for s in 1:t
            i=touched[s]; rows[nz+s]=i; vals[nz+s]=sums[i]
        end
        nz+=t; colptr[j+1]=nz+1
    end
    resize!(rows,nz); resize!(vals,nz)
    CSCMatrix(_Shared(),m,n,colptr,rows,vals)
end
# C = op(A)·X for a CSC matrix A and a dense X (k×n, possibly a transpose
# wrapper): op(A)=A scatters columns of A (C[:,j] += A[:,k]·X[k,j]); op(A)=Aᵀ
# gathers dot products (C[i,j] = A[:,i]·X[:,j]). Columns of C are split over
# threads for large products.
function _csc_mul(A::_CSCLike,transA::Bool,X::AbstractMatrix;threaded=true)
    m,n0,cp,ri,v=_csc_parts(A); T=promote_type(eltype(v),eltype(X))
    rows=transA ? n0 : m; n=size(X,2)
    C=zeros(T,rows,n)
    work=Float64(length(v))*n
    workers=_operation_workers(n,work,threaded,262144)
    _foreach_range(n,workers) do cols,_
        @inbounds for j in cols
            if transA
                for i in 1:n0
                    s=zero(T)
                    for p in cp[i]:cp[i+1]-1; s+=v[p]*X[ri[p],j]; end
                    C[i,j]=s
                end
            else
                for k in 1:n0
                    x=X[k,j]; iszero(x) && continue
                    for p in cp[k]:cp[k+1]-1; C[ri[p],j]+=v[p]*x; end
                end
            end
        end
    end
    C
end
# Matrix-product step: two 2-D operands with unique labels, one contracted
# label, output = the two free labels in either order. Returns the result in
# `out` order, or nothing when the step is not of this form.
function _matrix_pair(A,B,la,lb,out,dims;threaded=true)
    (ndims(A)==2 && ndims(B)==2 && length(la)==2 && length(lb)==2 && allunique(la) && allunique(lb) && length(out)==2) || return nothing
    shared=[l for l in la if l in lb]
    length(shared)==1 || return nothing
    k=shared[1]; k in out && return nothing
    i=la[1]==k ? la[2] : la[1]; j=lb[1]==k ? lb[2] : lb[1]
    (i in out && j in out && i!=j) || return nothing
    (size(A,1)==dims[la[1]] && size(A,2)==dims[la[2]] && size(B,1)==dims[lb[1]] && size(B,2)==dims[lb[2]]) || return nothing
    sa=A isa _CSCLike; sb=B isa _CSCLike
    if sa && sb
        # Orientation C = A·B (rows i) or Cᵀ = Bᵀ·Aᵀ (rows j): the one needing
        # the fewest O(nnz) transposes of operands and result.
        c1=(la[2]!=k)+(lb[1]!=k)+(out!=[i,j]); c2=(lb[2]!=k)+(la[1]!=k)+(out!=[j,i])
        if c1<=c2
            X=la[2]==k ? A : _csc_transpose(A); Y=lb[1]==k ? B : _csc_transpose(B)
            C=_csc_spgemm(X,Y); return out==[i,j] ? C : _csc_transpose(C)
        else
            X=lb[2]==k ? B : _csc_transpose(B); Y=la[1]==k ? A : _csc_transpose(A)
            C=_csc_spgemm(X,Y); return out==[j,i] ? C : _csc_transpose(C)
        end
    end
    # Sparse × dense in either order: OxiBLAS.mul's native kernels, with
    # transpose wrappers for the operand orientation and the output order.
    ((sa && B isa AbstractMatrix && !_is_sparse_operand(B)) || (sb && A isa AbstractMatrix && !_is_sparse_operand(A))) || return nothing
    # The product's left operand: A for out == [i, j], Bᵀ for out == [j, i].
    # A sparse left operand uses the CSC kernel above; a dense one uses
    # OxiBLAS.mul's dense × sparse kernel.
    if out==[i,j]
        sa && return _csc_mul(A,la[2]!=k,lb[1]==k ? B : transpose(B);threaded)
        return mul(A isa AbstractMatrix && la[2]==k ? A : transpose(A),lb[1]==k ? B : transpose(B))
    end
    sb && return _csc_mul(B,lb[2]!=k,la[1]==k ? A : transpose(A);threaded)
    mul(lb[2]==k ? B : transpose(B),la[1]==k ? A : transpose(A))
end
# 2-D unary steps on a CSC matrix: trace, diagonal, row/column sums, total
# sum, transpose. Returns (SparseTensor, labels) or nothing.
function _csc_unary(A::_CSCLike,labels,out)
    m,n,cp,ri,v=_csc_parts(A); T=eltype(v)
    if length(labels)==2 && labels[1]==labels[2]
        isempty(out) || out==[labels[1]] || return nothing
        idx=Int[]; vals=T[]
        @inbounds for j in 1:min(m,n)   # rows sorted per column: binary search
            lo=Int(cp[j]); hi=Int(cp[j+1])-1
            lo<=hi || continue
            p=searchsortedfirst(view(ri,lo:hi),j)+lo-1
            p<=hi && ri[p]==j && (push!(idx,j); push!(vals,v[p]))
        end
        if isempty(out)
            return SparseTensor{T,0}(_Shared(),(),isempty(idx) ? Int[] : [1],isempty(idx) ? T[] : [sum(vals)]),out
        end
        return SparseTensor{T,1}(_Shared(),(m,),idx,vals),out
    end
    (length(labels)==2 && allunique(labels)) || return nothing
    if isempty(out)
        nz=cp[n+1]-1
        return SparseTensor{T,0}(_Shared(),(),nz==0 ? Int[] : [1],nz==0 ? T[] : [sum(view(v,1:nz))]),out
    elseif out==[labels[1]]     # row sums (rows that hold entries)
        acc=zeros(T,m); seen=falses(m)
        @inbounds for p in 1:cp[n+1]-1; acc[ri[p]]+=v[p]; seen[ri[p]]=true; end
        idx=findall(seen); return SparseTensor{T,1}(_Shared(),(m,),idx,acc[idx]),out
    elseif out==[labels[2]]     # column sums (non-empty columns)
        idx=Int[]; vals=T[]
        @inbounds for j in 1:n
            cp[j]<cp[j+1] || continue
            s=zero(T); for p in cp[j]:cp[j+1]-1; s+=v[p]; end
            push!(idx,j); push!(vals,s)
        end
        return SparseTensor{T,1}(_Shared(),(n,),idx,vals),out
    elseif out==[labels[2],labels[1]]
        return SparseTensor(_csc_transpose(A)),out
    end
    nothing
end

# One pairwise step on any mix of operands; `keep` holds the labels needed
# after the step (output and remaining operands).
function _einsum_pair_any(A,B,la,lb,out,dims;threaded=true,allocator=nothing,kwargs...)
    if A isa _CSCLike || B isa _CSCLike
        R=_matrix_pair(A,B,la,lb,out,dims;threaded)
        R===nothing || return R
        A=_as_sparse_tensor(A); B=_as_sparse_tensor(B)
    end
    sa=A isa SparseTensor; sb=B isa SparseTensor
    !sa && !sb && return _einsum_pair(A,B,la,lb,out,dims;threaded,allocator,kwargs...)
    keepa=union(out,lb); keepb=union(out,la)
    A,la=sa ? _sparse_unary(A,la,keepa,dims) : _dense_unary(A,la,keepa,dims;threaded)
    B,lb=sb ? _sparse_unary(B,lb,keepb,dims) : _dense_unary(B,lb,keepb,dims;threaded)
    # A label carried by neither operand any more (broadcast on both) cannot
    # be an output label of a pair; the planner never requests it.
    sa && sb && return _sparse_pair(A,B,la,lb,out,dims)
    sa ? _sparse_dense_pair(A,B,la,lb,out,dims;threaded,allocator) : _sparse_dense_pair(B,A,lb,la,out,dims;threaded,allocator)
end

# Sparse-aware execution of a plan. Sparse operands become SparseTensor; a
# direct (unpaired) plan is evaluated left to right.
function _einsum_compute_sparse(plan,arrays,T;threaded=true,allocator=nothing,release=nothing,reuse_result=false,output=:auto,kwargs...)
    final_allocator=reuse_result ? allocator : nothing
    data=Any[]; ls=copy(plan.labels)
    for (k,A) in enumerate(arrays)
        if A isa _CSCLike
            push!(data,A)
        elseif A isa SparseTensor{<:Any,2}   # its linear index is CSC order: O(nnz) to CSCMatrix
            push!(data,CSCMatrix(A))
        elseif A isa CSRMatrix   # CSR of (i,j) = CSC of the transpose, labels (j,i)
            push!(data,CSCMatrix(_Shared(),A.n,A.m,A.rowptr,A.colind,A.values)); ls[k]=reverse(ls[k])
        else
            push!(data,_is_sparse_operand(A) ? _sparse_operand(A) : A)
        end
    end
    # Ellipsis broadcasting: a singleton axis facing a longer one elsewhere
    # carries no index; drop it (free for both kinds: removing size-1 axes
    # keeps a SparseTensor's linear indices) so joins match on real labels.
    for k in eachindex(data)
        # lk, not l: `l` is reassigned after this loop and was captured here.
        lk=ls[k]; keep=[size(data[k],d)!=1 || plan.dimensions[lk[d]]==1 for d in eachindex(lk)]
        all(keep) && continue
        data[k]=_as_sparse_tensor(data[k])
        data[k]=reshape(data[k],Tuple(size(data[k],d) for d in eachindex(lk) if keep[d])); ls[k]=lk[keep]
    end
    steps=plan.steps
    if plan.direct && length(data)>1
        steps=Tuple{Int,Int,Vector{Int}}[]; work=copy(ls)
        while length(work)>1
            needed=Set(plan.output); foreach(w->union!(needed,w),work[3:end])
            o=filter(in(needed),unique(vcat(work[1],work[2])))
            push!(steps,(1,2,o)); work=vcat(work[3:end],[o])
        end
    end
    final_order=allunique(plan.output)
    for (step,(i,j,o)) in enumerate(steps)
        pair_allocator=step==length(steps) ? final_allocator : allocator
        # Last step straight into the requested output order (same labels),
        # instead of a unary permutation of the whole result afterwards.
        step==length(steps) && final_order && length(o)==length(plan.output) && issetequal(o,plan.output) && (o=plan.output)
        C=_einsum_pair_any(data[i],data[j],ls[i],ls[j],o,plan.dimensions;threaded,allocator=pair_allocator,kwargs...)
        if release!==nothing && !(data[i] isa Union{SparseTensor,_CSCLike} && data[j] isa Union{SparseTensor,_CSCLike})
            release(data[i]); release(data[j])
        end
        deleteat!(data,j); deleteat!(data,i); push!(data,C)
        deleteat!(ls,j); deleteat!(ls,i); push!(ls,o)
    end
    R,l=data[1],ls[1]
    if !(l==plan.output && allunique(l)) && R isa _CSCLike
        U=_csc_unary(R,l,plan.output)
        U===nothing ? (R=SparseTensor(R)) : ((R,l)=U)
    end
    if !(l==plan.output && allunique(l))
        R,l=R isa SparseTensor ? _sparse_unary(R,l,plan.output,plan.dimensions) : _dense_unary(R,l,plan.output,plan.dimensions;threaded)
        if l!=plan.output
            # Unary result in first-occurrence order: permute to the output.
            # lu is bound once: the generator captured the reassigned `l`.
            lu=l; perm=Tuple(findfirst(==(x),lu)::Int for x in plan.output)
            R=permutedims(R,perm)
        end
    elseif length(arrays)==1 && !(R isa _CSCLike)
        R=copy(R)   # never return (a wrapper of) the caller's operand
    end
    # A CSC result: new storage unless it is the caller's own operand (a
    # single-operand plan with nothing to do); converted to SparseTensor unless
    # a sparse destination takes it natively.
    if R isa _CSCLike
        length(arrays)==1 && R===arrays[1] && (R=SparseTensor(R))
        output===:native || R isa SparseTensor || (R=SparseTensor(R))
    end
    _einsum_output(R,output,T)
end
# `output`: :auto keeps a sparse result sparse (all operands sparse) and a
# dense one dense; :dense and :sparse force the kind. Scalars (no output
# labels) are zero-dimensional Arrays unless :sparse is requested.
function _einsum_output(R,output,T)
    # :native (internal, sparse destinations of einsum!): a CSC result of the
    # matrix path is handed over as is, without the SparseTensor round trip.
    output===:native && return R isa _CSCLike ? R : _einsum_output(R,:sparse,T)
    output in (:auto,:dense,:sparse) || throw(ArgumentError("output must be :auto, :dense or :sparse"))
    if R isa SparseTensor
        (output===:dense || (output===:auto && ndims(R)==0)) && return Array(R)
        return R
    end
    output===:sparse ? SparseTensor(R) : R
end

# ---- Sparse destinations of einsum!/contract! ----------------------------

const SparseDestination=Union{SparseTensor,SparseMatrixCSC,SparseVector,CSRMatrix,CSCMatrix}
_as_tensor(C::SparseTensor)=C
_as_tensor(C)=SparseTensor(C)
# C <- alpha*R + beta*C with R sparse; C's storage vectors are rewritten in place.
function _sparse_axpby!(C,R::SparseTensor{T},alpha,beta) where T
    size(C)==size(R) || throw(DimensionMismatch("sparse output shape differs"))
    old=iszero(beta) ? SparseTensor(size(C),Int[],T[]) : _as_tensor(C)
    idx=Int[]; vals=T[]; p=1; q=1
    while p<=length(R.indices) || q<=length(old.indices)
        if q>length(old.indices) || (p<=length(R.indices) && R.indices[p]<old.indices[q])
            push!(idx,R.indices[p]); push!(vals,alpha*R.values[p]); p+=1
        elseif p>length(R.indices) || old.indices[q]<R.indices[p]
            push!(idx,old.indices[q]); push!(vals,beta*old.values[q]); q+=1
        else
            push!(idx,R.indices[p]); push!(vals,alpha*R.values[p]+beta*old.values[q]); p+=1; q+=1
        end
    end
    _sparse_store!(C,SparseTensor(size(C),idx,vals))
end
function _sparse_store!(C::SparseTensor,S::SparseTensor)
    resize!(C.indices,length(S.indices)); copyto!(C.indices,S.indices)
    resize!(C.values,length(S.values)); copyto!(C.values,S.values); C
end
function _sparse_store!(C::SparseVector,S::SparseTensor)
    nz=getfield(C,:nzind); v=SparseArrays.nonzeros(C)
    resize!(nz,length(S.indices)); copyto!(nz,S.indices); resize!(v,length(S.values)); copyto!(v,S.values); C
end
function _sparse_store!(C::Union{SparseMatrixCSC,CSCMatrix},S::SparseTensor)
    X=CSCMatrix(S)
    colptr,rows,vals=C isa SparseMatrixCSC ? (SparseArrays.getcolptr(C),SparseArrays.rowvals(C),SparseArrays.nonzeros(C)) : (C.colptr,C.rowind,C.values)
    copyto!(colptr,X.colptr); resize!(rows,length(X.rowind)); copyto!(rows,X.rowind); resize!(vals,length(X.values)); copyto!(vals,X.values); C
end
function _sparse_store!(C::CSRMatrix,S::SparseTensor)
    X=CSRMatrix(CSCMatrix(S))
    copyto!(C.rowptr,X.rowptr); resize!(C.colind,length(X.colind)); copyto!(C.colind,X.colind); resize!(C.values,length(X.values)); copyto!(C.values,X.values); C
end

# ---- contract / einsum! entry points for sparse operands and destinations ----

# Axis labels: A's axes 1..ndims(A); B's contracted axes reuse A's labels and
# its free axes get new ones. Output: A's free axes, then B's.
function _contract_sparse(A,B,ca,cb,fa,fb,conjugateA,conjugateB,output;threaded=true,kwargs...)
    T=eltype(A)
    la=collect(1:ndims(A)); lb=Vector{Int}(undef,ndims(B))
    for (j,d) in enumerate(cb)
        lb[d]=ca[j]
    end
    for (k,d) in enumerate(fb)
        lb[d]=ndims(A)+k
    end
    dims=Dict{Int,Int}()
    for d in 1:ndims(A); dims[la[d]]=size(A,d); end
    for d in 1:ndims(B); dims[lb[d]]=size(B,d); end
    out=vcat(la[fa],lb[fb])
    prepare(X,c)=_is_sparse_operand(X) ? (c ? conj(_sparse_operand(X)) : _sparse_operand(X)) : (c ? conj.(X) : X)
    R=_einsum_pair_any(prepare(A,conjugateA),prepare(B,conjugateB),la,lb,out,dims;threaded,kwargs...)
    _einsum_output(R,output,T)
end
"""einsum! into a sparse destination (SparseTensor, SparseMatrixCSC,
SparseVector, CSRMatrix, CSCMatrix): C ← α·result + β·C; C's storage vectors
are resized in place. The full result is formed before C changes."""
function _sparse_einsum!(C,plan::EinsumPlan,arrays,α,β;workspace=nothing,kwargs...)
    size(C)==Tuple(plan.dimensions[l] for l in plan.output) || throw(DimensionMismatch("einsum output shape differs"))
    T=eltype(C); all(A->eltype(A)===T,arrays) || throw(ArgumentError("einsum element types must agree"))
    α,β=convert(T,α),convert(T,β)
    iszero(α) && return _sparse_axpby!(C,SparseTensor(size(C),Int[],T[]),α,β)
    native=any(_is_sparse_operand,arrays)
    R=einsum(plan,arrays...;workspace,output=native ? :native : :sparse,kwargs...)
    # β = 0, α = 1 and a CSC result going into a CSC destination: copy the
    # arrays (no merge pass, no SparseTensor round trip).
    if iszero(β) && isone(α) && R isa _CSCLike && C isa Union{SparseMatrixCSC,CSCMatrix}
        m,n,cp,ri,v=_csc_parts(R)
        colptr,rows,vals=C isa SparseMatrixCSC ? (SparseArrays.getcolptr(C),SparseArrays.rowvals(C),SparseArrays.nonzeros(C)) : (C.colptr,C.rowind,C.values)
        copyto!(colptr,cp); resize!(rows,length(ri)); copyto!(rows,ri); resize!(vals,length(v)); copyto!(vals,v)
        return C
    end
    _sparse_axpby!(C,_as_tensor(R),α,β)
end
# One method per destination type (element type constrained) so that each is
# strictly more specific than the dense AbstractArray{T} methods.
for D in (:SparseTensor,:SparseMatrixCSC,:SparseVector,:CSRMatrix,:CSCMatrix)
    @eval begin
        einsum!(C::$D{T},plan::EinsumPlan,arrays::AbstractArray...;α=one(T),β=zero(T),kwargs...) where {T<:GenericFloat}=_sparse_einsum!(C,plan,arrays,α,β;kwargs...)
        function einsum!(C::$D{T},plan::SlicedEinsumPlan,arrays::AbstractArray...;α=one(T),β=zero(T),kwargs...) where {T<:GenericFloat}
            _sparse_axpby!(C,SparseTensor(einsum(plan,arrays...;kwargs...)),convert(T,α),convert(T,β))
        end
        function contract!(C::$D{T},A::AbstractArray{T},B::AbstractArray{T},axesA,axesB;α=one(T),β=zero(T),kwargs...) where {T<:GenericFloat}
            R=contract(A,B,axesA,axesB;output=:sparse,kwargs...)
            _sparse_axpby!(C,_as_tensor(R),convert(T,α),convert(T,β))
        end
    end
end

# Tiles of sliced plans: a sparse operand is restricted to the tile's index
# ranges (entries filtered and renumbered), not viewed element by element.
_einsum_tile(A,indices)=view(A,indices...)
function _einsum_tile(A::SparseTensor{T,N},indices) where {T,N}
    ranges=ntuple(d->indices[d] isa Colon ? (1:A.dims[d]) : indices[d],N)
    dims=map(length,ranges); L=LinearIndices(dims); keys=Int[]; vals=T[]
    for p in eachindex(A.indices)
        c=_sparse_coords(A.indices[p],A.dims)
        all(d->c[d] in ranges[d],1:N) || continue
        push!(keys,L[ntuple(d->c[d]-first(ranges[d])+1,N)...]); push!(vals,A.values[p])
    end
    SparseTensor(dims,keys,vals)   # filtering preserves the column-major order
end
