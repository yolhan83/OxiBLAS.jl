# SPDX-License-Identifier: Apache-2.0
# Structural and elementwise Base/LinearAlgebra operations on the OxiBLAS
# sparse types (matrix formats and SparseTensor of any order): indexing,
# broadcast/map, reductions, comparisons, concatenation, triangles... in
# O(nnz) or O(nnz log nnz), through the canonical coordinate form of
# SparseTensor (sorted column-major linear indices). No SparseArrays or
# LinearAlgebra kernel and no external BLAS is used. Every method has an
# OxiBLAS type in its signature (no type piracy).
#
# Result formats keep the operand's family: CSCMatrix → CSCMatrix,
# COOMatrix → COOMatrix, SparseTensor → SparseTensor, CSRMatrix and the
# block/padded formats (DIA, ELL, BSR, BSC, HYB, SELL) → CSRMatrix (a
# structural result is not rebuilt in a padded or blocked layout). A
# one-dimensional sparse result is a SparseTensor of order 1 (an
# AbstractSparseVector).

const _OxiAll=Union{_OxiSparse,SparseTensor}
_coo(A::SparseTensor)=A
_coo(A::Union{_OxiSparse,SparseMatrixCSC,SparseVector})=SparseTensor(A)
_like(::CSCMatrix,T::SparseTensor{<:Any,2})=CSCMatrix(T)
_like(::COOMatrix,T::SparseTensor{<:Any,2})=COOMatrix(T)
_like(::SparseTensor,T::SparseTensor)=T
_like(::SparseTensor,T::SparseTensor{<:Any,2})=T
_like(::Any,T::SparseTensor{<:Any,2})=CSRMatrix(T)
_like(::Any,T::SparseTensor)=T          # a result of another order stays a tensor
_tensor_keys(dims,coords)=LinearIndices(dims)[coords...]

# ---- indexing -------------------------------------------------------------------------
# Position lookup of one index set: stored coordinate c → output positions.
_index_positions(s::Colon,c)=(c,)
_index_positions(s::Integer,c)=c==s ? (1,) : ()
_index_positions(s::AbstractUnitRange{<:Integer},c)=first(s)<=c<=last(s) ? (c-first(s)+1,) : ()
_index_positions(s::Dict{Int,Vector{Int}},c)=get(s,c,())
function _index_table(s,n)
    s isa Union{Colon,Integer,AbstractUnitRange{<:Integer}} && return s
    t=Dict{Int,Vector{Int}}()
    for (k,c) in enumerate(s)
        1<=c<=n || throw(BoundsError())
        push!(get!(()->Int[],t,Int(c)),k)
    end
    t
end
_index_length(s::Colon,n)=n
_index_length(s,n)=length(s)
"""
    _sparse_getindex(A::SparseTensor{T,N}, I::Tuple) -> SparseTensor

A[I...] with integers, ranges, vectors and colons on a SparseTensor:
stored entries are mapped through the selected positions (an index may
repeat; an integer index drops its dimension).
"""
function _sparse_getindex(A::SparseTensor{T,N},I::Tuple) where {T,N}
    length(I)==N || throw(ArgumentError("expected $N indices"))
    for d in 1:N
        I[d] isa Colon || checkbounds(Bool,1:A.dims[d],I[d]) || throw(BoundsError(A,I))
    end
    tables=ntuple(d->_index_table(I[d],A.dims[d]),N)
    kept=Tuple(d for d in 1:N if !(I[d] isa Integer))
    outdims=Tuple(_index_length(I[d],A.dims[d]) for d in kept)
    keys=Int[]; vals=T[]
    L=LinearIndices(outdims)
    for p in eachindex(A.indices)
        c=_sparse_coords(A.indices[p],A.dims)
        pos=ntuple(d->_index_positions(tables[d],c[d]),N)
        any(isempty,pos) && continue
        for q in Iterators.product((pos[d] for d in kept)...)
            push!(keys,isempty(q) ? 1 : L[q...]); push!(vals,A.values[p])
        end
    end
    _sparse_tensor_from(outdims,keys,vals)
end
const _SparseIndex=Union{Integer,Colon,AbstractVector{<:Integer}}
# Matrix formats: any non-scalar index pair.
function _matrix_getindex(A,i,j)
    R=_sparse_getindex(_coo(A),(i,j))
    ndims(R)==2 ? _like(A,R) : R
end
Base.getindex(A::_OxiSparse,i::Union{Colon,AbstractVector{<:Integer}},j::Union{Colon,AbstractVector{<:Integer}})=_matrix_getindex(A,i,j)
Base.getindex(A::_OxiSparse,i::Integer,j::Union{Colon,AbstractVector{<:Integer}})=_matrix_getindex(A,i,j)
Base.getindex(A::_OxiSparse,i::Union{Colon,AbstractVector{<:Integer}},j::Integer)=_matrix_getindex(A,i,j)
# Order 1: one index (the method set mirrors SparseArrays' vector indexing,
# which an order-1 tensor would otherwise reach).
Base.getindex(A::SparseTensor{<:Any,1},I::AbstractVector{<:Integer})=_sparse_getindex(A,(I,))
Base.getindex(A::SparseTensor{<:Any,1},::Colon)=copy(A)
Base.getindex(A::SparseTensor{<:Any,1},I::AbstractUnitRange{<:Integer})=_sparse_getindex(A,(I,))
Base.getindex(A::SparseTensor{<:Any,1},I::AbstractVector{Bool})=_sparse_getindex(A,(findall(I),))
# Order ≥ 2: one index per dimension (other index counts use Base's rules).
function Base.getindex(A::SparseTensor{T,N},i::_SparseIndex,j::_SparseIndex,I::_SparseIndex...) where {T,N}
    J=(i,j,I...)
    length(J)==N || return invoke(getindex,Tuple{AbstractArray,Vararg{Any}},A,J...)
    all(k->k isa Integer,J) && return A[LinearIndices(A.dims)[J...]]
    _sparse_getindex(A,J)
end

# ---- setindex! (insertion keeps the format) ---------------------------------------------
# Assigning to a stored entry overwrites it (an explicit zero included, as in
# SparseMatrixCSC); assigning zero to an absent entry stores nothing.
function Base.setindex!(A::CSCMatrix,v,i::Integer,j::Integer)
    @boundscheck checkbounds(A,i,j)
    lo,hi=A.colptr[j],A.colptr[j+1]-1
    p=lo-1+searchsortedfirst(view(A.rowind,lo:hi),i)
    if p<=hi && A.rowind[p]==i
        A.values[p]=v
    elseif !iszero(v)
        insert!(A.rowind,p,i); insert!(A.values,p,v)
        for k in j+1:A.n+1
            A.colptr[k]+=1
        end
    end
    A
end
function Base.setindex!(A::CSRMatrix,v,i::Integer,j::Integer)
    @boundscheck checkbounds(A,i,j)
    lo,hi=A.rowptr[i],A.rowptr[i+1]-1
    for p in lo:hi
        A.colind[p]==j && (A.values[p]=v; return A)
    end
    iszero(v) && return A
    insert!(A.colind,hi+1,j); insert!(A.values,hi+1,v)
    for k in i+1:A.m+1
        A.rowptr[k]+=1
    end
    A
end
# Canonical COO (sorted by (row, column), no stored zeros): zero removes.
function Base.setindex!(A::COOMatrix,v,i::Integer,j::Integer)
    @boundscheck checkbounds(A,i,j)
    lo,hi=1,length(A.values)
    while lo<=hi
        mid=(lo+hi)>>>1
        (A.rows[mid],A.cols[mid])<(i,j) ? (lo=mid+1) : (hi=mid-1)
    end
    if lo<=length(A.values) && A.rows[lo]==i && A.cols[lo]==j
        iszero(v) ? (deleteat!(A.rows,lo); deleteat!(A.cols,lo); deleteat!(A.values,lo)) : (A.values[lo]=v)
    elseif !iszero(v)
        insert!(A.rows,lo,i); insert!(A.cols,lo,j); insert!(A.values,lo,v)
    end
    A
end

# ---- broadcast and map -----------------------------------------------------------------
"""
    OxiSparseStyle{N} <: Broadcast.AbstractArrayStyle{N}

Broadcast style of the OxiBLAS sparse types. Same-shape sparse operands
with scalars: a function with f(0,…,0)==0 gives a sparse result over the
union of the stored patterns (computed zeros dropped, as SparseArrays does);
any other function gives a dense result filled with f(0,…,0). Dense
operands or size-1 broadcasting give a dense result (the sparse operands are
expanded by OxiBLAS's own conversion).
"""
struct OxiSparseStyle{N} <: Broadcast.AbstractArrayStyle{N} end
OxiSparseStyle{N}(::Val{M}) where {N,M}=OxiSparseStyle{M}()
Broadcast.BroadcastStyle(::Type{<:_OxiSparse})=OxiSparseStyle{2}()
Broadcast.BroadcastStyle(::Type{<:SparseTensor{<:Any,N}}) where N=OxiSparseStyle{N}()
Broadcast.BroadcastStyle(::OxiSparseStyle{M},::Broadcast.DefaultArrayStyle{N}) where {M,N}=OxiSparseStyle{max(M,N)}()
Broadcast.BroadcastStyle(::OxiSparseStyle{M},::SparseArrays.HigherOrderFns.SparseMatStyle) where M=OxiSparseStyle{max(M,2)}()
Broadcast.BroadcastStyle(::OxiSparseStyle{M},::SparseArrays.HigherOrderFns.SparseVecStyle) where M=OxiSparseStyle{max(M,1)}()
Broadcast.BroadcastStyle(::OxiSparseStyle{M},::SparseArrays.HigherOrderFns.PromoteToSparse) where M=OxiSparseStyle{max(M,2)}()
_bcast_sparse_arg(a)=a isa Union{_OxiAll,SparseMatrixCSC,SparseVector}
_bcast_scalar(a)=a isa Number || a isa Base.RefValue || a isa Ref
_bcast_value(a)=a isa Number ? a : a[]
function Base.copy(bc::Broadcast.Broadcasted{<:OxiSparseStyle})
    fb=Broadcast.flatten(bc); f=fb.f; args=fb.args
    dims=map(length,axes(bc))
    if all(a->_bcast_scalar(a) || (_bcast_sparse_arg(a) && size(a)==dims),args)
        leader=args[findfirst(_bcast_sparse_arg,args)]
        return _bcast_sparse(f,args,dims,leader)
    end
    dense=map(a->_bcast_sparse_arg(a) ? Array(a) : a,args)
    broadcast(f,dense...)
end
# k-way merge of the stored patterns; f evaluated with zero for absent entries.
function _bcast_sparse(f,args,dims,leader)
    ts=map(a->_bcast_sparse_arg(a) ? _coo(a) : a,args)
    zeroargs=map(a->a isa SparseTensor ? zero(eltype(a)) : _bcast_value(a),ts)
    z=f(zeroargs...)
    idx=Int[]
    for a in ts
        a isa SparseTensor && append!(idx,a.indices)
    end
    sort!(idx); unique!(idx)
    # Merge cursors, mutated in place: rebinding a tuple of cursors inside
    # the ntuple closure was a Core.Box per output entry.
    ptr=ones(Int,length(ts))
    vals=Vector{Any}(undef,length(idx))
    @inbounds for (q,k) in enumerate(idx)
        x=ntuple(length(ts)) do r
            a=ts[r]
            a isa SparseTensor || return zeroargs[r]
            p=ptr[r]
            while p<=length(a.indices) && a.indices[p]<k
                p+=1
            end
            ptr[r]=p
            p<=length(a.indices) && a.indices[p]==k ? a.values[p] : zeroargs[r]
        end
        vals[q]=f(x...)
    end
    T=isempty(vals) ? typeof(z) : mapreduce(typeof,promote_type,vals;init=typeof(z))
    if iszero(z)
        keep=findall(!iszero,vals)
        R=SparseTensor(Tuple(dims),idx[keep],convert(Vector{T},vals[keep]))
        return length(dims)==ndims(leader) ? _like(leader,R) : R
    end
    D=fill(convert(T,z),Tuple(dims))
    for (q,k) in enumerate(idx)
        D[k]=vals[q]
    end
    D
end
Base.map(f,A::_OxiAll)=f.(A)
function _sparse_map(f,A,Bs...)
    all(B->size(B)==size(A),Bs) || throw(DimensionMismatch("map: dimensions differ"))
    broadcast(f,A,Bs...)
end
Base.map(f,A::_OxiAll,Bs::_OxiAll...)=_sparse_map(f,A,Bs...)
# SparseArrays maps +, -, *, min, max over any two AbstractSparseVectors.
for op in (+,-,*,min,max)
    Base.map(f::typeof(op),A::SparseTensor{<:Any,1},B::SparseTensor{<:Any,1})=_sparse_map(f,A,B)
end
# In-place broadcast into a dense destination, or into a mutable sparse one
# (CSR, CSC, COO, SparseTensor) whose storage is replaced by the result.
Base.copyto!(dest::StridedArray,bc::Broadcast.Broadcasted{<:OxiSparseStyle})=copyto!(dest,copy(bc))
function Base.copyto!(dest::Union{CSRMatrix,CSCMatrix,COOMatrix,SparseTensor},bc::Broadcast.Broadcasted{<:OxiSparseStyle})
    size(dest)==map(length,axes(bc)) || throw(DimensionMismatch("broadcast destination size"))
    R=copy(bc)
    _replace_storage!(dest,R isa _OxiAll ? _coo(R) : SparseTensor(R))
end
function _replace_storage!(dest,T::SparseTensor)
    S=dest isa SparseTensor ? T : dest isa CSCMatrix ? CSCMatrix(T) : dest isa CSRMatrix ? CSRMatrix(T) : COOMatrix(T)
    for f in fieldnames(typeof(dest))
        v=getfield(dest,f)
        v isa Vector && (resize!(v,length(getfield(S,f))); copyto!(v,getfield(S,f)))
    end
    dest
end
# real/imag/abs… of an array go through map or broadcast in Base, hence
# through OxiSparseStyle: no method is needed here.

# ---- dense conversion -------------------------------------------------------------------
Base.copyto!(dest::Vector,src::SparseTensor{<:Any,1})=_sparse_copyto!(dest,src)
Base.copyto!(dest::StridedArray,src::_OxiAll)=_sparse_copyto!(dest,src)
function _sparse_copyto!(dest,src)
    size(dest)==size(src) || length(dest)==length(src) || throw(DimensionMismatch("copyto!: sizes differ"))
    T=_coo(src); fill!(dest,zero(eltype(dest)))
    @inbounds for p in eachindex(T.indices)
        dest[T.indices[p]]=T.values[p]
    end
    dest
end
Base.Array(A::_OxiSparse)=Array(_coo(A))
Base.Matrix(A::_OxiSparse)=Array(_coo(A))

# ---- reductions -------------------------------------------------------------------------
# The implicit zeros of a reduction contribute f(0) once per missing entry;
# for + and * this is folded in closed form, for idempotent operators once.
const _IDEMPOTENT_OPS=(max,min,&,|)
function _fold_zeros(op,acc,z,k)
    k==0 && return acc
    c= op===Base.add_sum || op===(+) ? k*z :
       op===Base.mul_prod || op===(*) ? z^k :
       any(o->op===o,_IDEMPOTENT_OPS) ? z : nothing
    if c===nothing
        for _ in 1:k
            acc=acc isa Base._InitialValue ? z : op(acc,z)
        end
        return acc
    end
    acc isa Base._InitialValue ? c : op(acc,c)
end
function Base.mapreduce(f,op,A::_OxiAll;dims=:,init=Base._InitialValue())
    T=_coo(A)
    dims isa Colon || return _mapreduce_dims(f,op,T,A,dims,init)
    acc=init
    for v in T.values
        acc=acc isa Base._InitialValue ? f(v) : op(acc,f(v))
    end
    acc=_fold_zeros(op,acc,f(zero(eltype(T))),length(T)-length(T.values))
    acc isa Base._InitialValue ? Base.mapreduce_empty(f,op,eltype(T)) : acc
end
function _mapreduce_dims(f,op,T::SparseTensor{V,N},A,dims,init) where {V,N}
    red=Tuple(d for d in unique(dims) if 1<=d<=N)
    all(d->d>=1,dims) || throw(ArgumentError("region dimensions must be positive"))
    outdims=ntuple(d->d in red ? 1 : T.dims[d],N)
    extent=prod((T.dims[d] for d in red);init=1)
    z=f(zero(V))
    empty=_fold_zeros(op,init,z,extent)
    empty isa Base._InitialValue && (empty=Base.mapreduce_empty(f,op,V))
    L=LinearIndices(outdims)
    keys=[L[ntuple(d->d in red ? 1 : c[d],N)...] for c in (_sparse_coords(i,T.dims) for i in T.indices)]
    order=sortperm(keys)
    outk=Int[]; outv=Any[]
    p=1
    while p<=length(order)
        k=keys[order[p]]; acc=init; cnt=0
        while p<=length(order) && keys[order[p]]==k
            v=f(T.values[order[p]]); acc=acc isa Base._InitialValue ? v : op(acc,v); cnt+=1; p+=1
        end
        push!(outk,k); push!(outv,_fold_zeros(op,acc,z,extent-cnt))
    end
    R=isempty(outv) ? typeof(empty) : mapreduce(typeof,promote_type,outv;init=typeof(empty))
    # A tensor keeps a sparse result when absent groups reduce to zero.
    if T===A && iszero(empty)
        keep=findall(!iszero,outv)
        return SparseTensor(outdims,outk[keep],convert(Vector{R},outv[keep]))
    end
    D=fill(convert(R,empty),outdims)
    for (q,k) in enumerate(outk)
        D[k]=outv[q]
    end
    D
end
# count over the whole array is not a mapreduce in Base (it iterates).
function Base.count(f,A::_OxiAll;dims=:,init=0)
    dims isa Colon || return mapreduce(x->Bool(f(x)),Base.add_sum,A;dims,init)
    T=_coo(A); nz=length(T)-length(T.values)
    c=init+count(f,T.values)
    nz>0 && Bool(f(zero(eltype(T)))) ? c+nz : c
end
function Base.any(f::Function,A::_OxiAll;dims=:)
    dims isa Colon || return mapreduce(x->f(x)::Bool,|,A;dims)
    T=_coo(A)
    any(f,T.values) || (length(T)>length(T.values) && f(zero(eltype(T))))
end
function Base.all(f::Function,A::_OxiAll;dims=:)
    dims isa Colon || return mapreduce(x->f(x)::Bool,&,A;dims)
    T=_coo(A)
    all(f,T.values) && (length(T)==length(T.values) || f(zero(eltype(T))))
end

# ---- comparison and search ---------------------------------------------------------------
# Equal values at equal positions; stored zeros equal absent entries.
function _sparse_equal(A,B)
    size(A)==size(B) || return false
    S=SparseArrays.dropzeros(_coo(A)); T=SparseArrays.dropzeros(_coo(B))
    S.indices==T.indices && S.values==T.values
end
Base.:(==)(A::_OxiAll,B::_OxiAll)=_sparse_equal(A,B)
Base.:(==)(A::_OxiAll,B::Union{SparseMatrixCSC,SparseVector})=_sparse_equal(A,B)
Base.:(==)(A::Union{SparseMatrixCSC,SparseVector},B::_OxiAll)=_sparse_equal(A,B)
Base.findall(f::Function,A::_OxiAll)=_sparse_findall(f,A)
Base.findall(f::Base.Fix2{typeof(in)},A::_OxiAll)=_sparse_findall(f,A)
function _sparse_findall(f,A)
    T=_coo(A)
    f(zero(eltype(T))) && return invoke(findall,Tuple{Function,AbstractArray},f,A)
    sel=[T.indices[p] for p in eachindex(T.values) if f(T.values[p])]
    ndims(A)==1 ? sel : [CartesianIndices(T.dims)[i] for i in sel]
end
"""
    findnz(A::_OxiSparse) -> Tuple{Vector{Int}, Vector{Int}, Vector}

Stored entries (explicit zeros included) as row indices, column indices and
values, in column-major order.
"""
function SparseArrays.findnz(A::_OxiSparse)
    T=_coo(A); m=size(A,1)
    ([(i-1)%m+1 for i in T.indices],[(i-1)÷m+1 for i in T.indices],copy(T.values))
end
SparseArrays.nnz(A::ExtendedSparse)=nnz(CSRMatrix(A))
SparseArrays.nonzeros(A::Union{CSRMatrix,COOMatrix})=A.values

# ---- stored zeros and tolerances ---------------------------------------------------------
# Compact a compressed structure in place, keeping the entries keep(s,p).
function _compact!(ptr,ind,vals,keep)
    w=1
    for s in 1:length(ptr)-1
        lo,hi=ptr[s],ptr[s+1]-1; ptr[s]=w
        for p in lo:hi
            if keep(s,ind[p],vals[p])
                ind[w]=ind[p]; vals[w]=vals[p]; w+=1
            end
        end
    end
    ptr[end]=w; resize!(ind,w-1); resize!(vals,w-1)
end
_filter!(A::CSCMatrix,keep)=(_compact!(A.colptr,A.rowind,A.values,(j,i,v)->keep(i,j,v)); A)
_filter!(A::CSRMatrix,keep)=(_compact!(A.rowptr,A.colind,A.values,(i,j,v)->keep(i,j,v)); A)
function _filter!(A::COOMatrix,keep)
    del=[p for p in eachindex(A.values) if !keep(A.rows[p],A.cols[p],A.values[p])]
    deleteat!(A.rows,del); deleteat!(A.cols,del); deleteat!(A.values,del); A
end
function _filter!(A::SparseTensor,keep)
    del=[p for p in eachindex(A.values) if !keep(_sparse_coords(A.indices[p],A.dims)...,A.values[p])]
    deleteat!(A.indices,del); deleteat!(A.values,del); A
end
const _MutableSparse=Union{CSRMatrix,CSCMatrix,COOMatrix,SparseTensor}
SparseArrays.dropzeros!(A::_MutableSparse)=_filter!(A,(c...)->!iszero(c[end]))
SparseArrays.droptol!(A::_MutableSparse,tol::Real)=_filter!(A,(c...)->abs(c[end])>tol)
SparseArrays.dropzeros(A::_MutableSparse)=SparseArrays.dropzeros!(copy(A))
SparseArrays.dropzeros(A::ExtendedSparse)=SparseArrays.dropzeros!(CSRMatrix(A))

# ---- triangles, diagonal, symmetry ----------------------------------------------------------
_LA.triu!(A::_MutableSparse,k::Integer=0)=(ndims(A)==2 || throw(ArgumentError("triu! needs a matrix")); _filter!(A,(i,j,v)->j-i>=k))
_LA.tril!(A::_MutableSparse,k::Integer=0)=(ndims(A)==2 || throw(ArgumentError("tril! needs a matrix")); _filter!(A,(i,j,v)->j-i<=k))
_LA.triu(A::Union{_OxiSparse,SparseTensor{<:Any,2}},k::Integer=0)=_LA.triu!(_sparse_mutable_copy(A),k)
_LA.tril(A::Union{_OxiSparse,SparseTensor{<:Any,2}},k::Integer=0)=_LA.tril!(_sparse_mutable_copy(A),k)
_sparse_mutable_copy(A::_MutableSparse)=copy(A)
_sparse_mutable_copy(A)=CSRMatrix(A)
"""
    diag(A::Union{_OxiSparse, SparseTensor{<:Any,2}}, k::Integer=0) -> SparseTensor

Diagonal k as a SparseTensor of order 1 (an AbstractSparseVector).
"""
function _LA.diag(A::Union{_OxiSparse,SparseTensor{<:Any,2}},k::Integer=0)
    m,n=size(A); len=max(0,k>=0 ? min(m,n-k) : min(m+k,n))
    T=_coo(A); keys=Int[]; vals=eltype(A)[]
    for p in eachindex(T.indices)
        i,j=_sparse_coords(T.indices[p],T.dims)
        j-i==k && (push!(keys,k>=0 ? i : j); push!(vals,T.values[p]))
    end
    _sparse_tensor_from((len,),keys,vals)
end
_LA.tr(A::Union{_OxiSparse,SparseTensor{<:Any,2}})=(_LA.checksquare(A); sum(_LA.diag(A).values;init=zero(eltype(A))))
# O(nnz log d), no copy for CSCMatrix (one CSC conversion otherwise): each
# stored entry is compared with its mirror, found by binary search in the
# sorted mirror column (CSCMatrix invariant); a missing entry counts as zero,
# so stored zeros are ignored. Stops at the first mismatch, as SparseArrays.
function _sparse_symmetric(A,herm)
    size(A,1)==size(A,2) || return false
    S=A isa CSCMatrix ? A : CSCMatrix(A)
    cp,ri,v=S.colptr,S.rowind,S.values
    @inbounds for j in 1:S.n, q in cp[j]:cp[j+1]-1
        i=Int(ri[q]); x=v[q]
        if i==j
            w=x
        else
            lo,hi=Int(cp[i]),Int(cp[i+1])-1
            r=searchsortedfirst(view(ri,lo:hi),j)+lo-1
            w=r<=hi && ri[r]==j ? v[r] : zero(x)
        end
        x==(herm ? conj(w) : w) || return false
    end
    true
end
_LA.issymmetric(A::Union{_OxiSparse,SparseTensor{<:Any,2}})=_sparse_symmetric(A,false)
_LA.ishermitian(A::Union{_OxiSparse,SparseTensor{<:Any,2}})=_sparse_symmetric(A,true)

# ---- inner products ------------------------------------------------------------------------
# Frobenius dot(A,B)=Σ conj(a)b over the common stored positions.
function _sparse_dot(A,B)
    size(A)==size(B) || throw(DimensionMismatch("dot: dimensions differ"))
    S=_coo(A); T=_coo(B); s=zero(promote_type(eltype(A),eltype(B)))*false
    p=q=1
    while p<=length(S.indices) && q<=length(T.indices)
        a,b=S.indices[p],T.indices[q]
        if a==b
            s+=_LA.dot(S.values[p],T.values[q]); p+=1; q+=1
        else
            a<b ? (p+=1) : (q+=1)
        end
    end
    s
end
_LA.dot(A::_OxiAll,B::_OxiAll)=_sparse_dot(A,B)
_LA.dot(A::_OxiAll,B::Union{SparseMatrixCSC,SparseVector})=_sparse_dot(A,B)
_LA.dot(A::Union{SparseMatrixCSC,SparseVector},B::_OxiAll)=_sparse_dot(A,B)
# A sparse operand against a dense one: driven by the stored entries.
function _LA.dot(A::_OxiAll,B::StridedArray)
    size(A)==size(B) || throw(DimensionMismatch("dot: dimensions differ")); T=_coo(A)
    sum(p->_LA.dot(T.values[p],B[T.indices[p]]),eachindex(T.values);init=zero(promote_type(eltype(A),eltype(B))))
end
_LA.dot(A::StridedArray,B::_OxiAll)=conj(_LA.dot(B,A))
"""
    dot(x::AbstractVector, A::Union{_OxiSparse, SparseTensor{<:Any,2}}, y::AbstractVector) -> Number

dot(x,A,y)=Σ conj(x[i]) a_ij y[j] over the stored entries of A.
"""
function _LA.dot(x::AbstractVector,A::Union{_OxiSparse,SparseTensor{<:Any,2}},y::AbstractVector)
    (length(x),length(y))==size(A) || throw(DimensionMismatch("dot(x,A,y): dimensions differ"))
    T=_coo(A); s=zero(promote_type(eltype(x),eltype(A),eltype(y)))
    for p in eachindex(T.indices)
        i,j=_sparse_coords(T.indices[p],T.dims)
        s+=_LA.dot(x[i],T.values[p]*y[j])
    end
    s
end

# ---- Kronecker product, concatenation, reversal, reshape --------------------------------------
_kron_tensor(X::Union{_OxiAll,SparseMatrixCSC})=_coo(X)
_kron_tensor(X::AbstractMatrix)=SparseTensor(X)
function _sparse_kron(A,B,leader)
    S=_kron_tensor(A); T=_kron_tensor(B)
    (ndims(S),ndims(T))==(2,2) || throw(ArgumentError("kron: matrices expected"))
    (m,n),(p,q)=S.dims,T.dims; dims=(m*p,n*q)
    keys=Vector{Int}(undef,length(S.values)*length(T.values)); vals=Vector{promote_type(eltype(S),eltype(T))}(undef,length(keys))
    w=0; L=LinearIndices(dims)
    for a in eachindex(S.indices)
        i,j=_sparse_coords(S.indices[a],S.dims)
        for b in eachindex(T.indices)
            k,l=_sparse_coords(T.indices[b],T.dims)
            w+=1; keys[w]=L[(i-1)*p+k,(j-1)*q+l]; vals[w]=S.values[a]*T.values[b]
        end
    end
    R=_sparse_tensor_from(dims,keys,vals)
    leader isa CSCMatrix && B isa CSCMatrix ? CSCMatrix(R) : leader isa SparseTensor ? R : CSRMatrix(R)
end
Base.kron(A::Union{_OxiSparse,SparseTensor{<:Any,2}},B::AbstractMatrix)=_sparse_kron(A,B,A)
Base.kron(A::StridedMatrix,B::Union{_OxiSparse,SparseTensor{<:Any,2}})=_sparse_kron(A,B,B)
Base.kron(A::SparseMatrixCSC,B::Union{_OxiSparse,SparseTensor{<:Any,2}})=_sparse_kron(A,B,B)
# Concatenation along dimension d of sparse and dense pieces.
_cat_tensor(X::Union{_OxiAll,SparseMatrixCSC,SparseVector},nd)=_cat_reshape(_coo(X),nd)
_cat_tensor(X::AbstractArray,nd)=_cat_reshape(SparseTensor(X),nd)
_cat_reshape(T::SparseTensor,nd)=ndims(T)>=nd ? T : reshape(T,(size(T)...,ntuple(_->1,nd-ndims(T))...))
function _sparse_cat(d,pieces)
    nd=max(d,maximum(ndims,pieces))
    ts=[_cat_tensor(X,nd) for X in pieces]
    for t in ts, e in 1:nd
        e==d || size(t,e)==size(ts[1],e) || throw(DimensionMismatch("cat: dimension $e differs"))
    end
    dims=ntuple(e->e==d ? sum(t->size(t,d),ts) : size(ts[1],e),nd)
    L=LinearIndices(dims); keys=Int[]; vals=promote_type(map(eltype,ts)...)[]; off=0
    for t in ts
        o=off   # per-piece binding: the ntuple closure captured the running `off`
        for p in eachindex(t.indices)
            c=_sparse_coords(t.indices[p],t.dims)
            push!(keys,L[ntuple(e->e==d ? c[e]+o : c[e],nd)...]); push!(vals,t.values[p])
        end
        off+=size(t,d)
    end
    R=SparseTensor(dims,keys[sortperm(keys)],vals[sortperm(keys)])
    nd!=2 ? R : all(X->X isa CSCMatrix || !(X isa _OxiAll),pieces) ? CSCMatrix(R) : CSRMatrix(R)
end
# Element types are spelled `<:Number` (every OxiBLAS sparse eltype is a
# number): SparseArrays 1.10 defines hcat/vcat on
# Union{Number,AbstractVecOrMat{<:Number}}..., and only a signature inside
# that union is unambiguous with it. Dispatch only, no runtime cost.
const _OxiCatNum=Union{CSRMatrix{<:Number},CSCMatrix{<:Number},COOMatrix{<:Number},DIAMatrix{<:Number},
                       ELLMatrix{<:Number},BSRMatrix{<:Number},BSCMatrix{<:Number},HYBMatrix{<:Number},
                       SELLMatrix{<:Number},SparseTensor{<:Number,1},SparseTensor{<:Number,2}}
Base.hcat(A::_OxiCatNum,Bs::AbstractVecOrMat{<:Number}...)=_sparse_cat(2,(A,Bs...))
# SparseArrays concatenates any AbstractSparseVectors (and dense vectors).
Base.hcat(A::SparseTensor{<:Number,1},Bs::SparseArrays.AbstractSparseVector{<:Number}...)=_sparse_cat(2,(A,Bs...))
Base.hcat(A::SparseTensor{<:Number,1},Bs::Union{Vector{<:Number},SparseArrays.AbstractSparseVector{<:Number}}...)=_sparse_cat(2,(A,Bs...))
Base.vcat(A::Union{CSRMatrix{<:Number},CSCMatrix{<:Number},COOMatrix{<:Number},DIAMatrix{<:Number},
                   ELLMatrix{<:Number},BSRMatrix{<:Number},BSCMatrix{<:Number},HYBMatrix{<:Number},
                   SELLMatrix{<:Number},SparseTensor{<:Number,2}},Bs::AbstractVecOrMat{<:Number}...)=_sparse_cat(1,(A,Bs...))
Base.vcat(A::SparseTensor{<:Number,1},Bs::SparseTensor{<:Number,1}...)=_sparse_cat(1,(A,Bs...))
Base.cat(A::SparseTensor,Bs::SparseTensor...;dims::Integer)=_sparse_cat(dims,(A,Bs...))
function Base.dropdims(A::SparseTensor;dims)
    all(d->1<=d<=ndims(A) && size(A,d)==1,dims) || throw(ArgumentError("dropped dims must be in range and have length 1"))
    reshape(A,Tuple(size(A,d) for d in 1:ndims(A) if !(d in dims)))
end
function Base.reverse(A::_OxiAll;dims=:)
    T=_coo(A); N=ndims(T); rd=dims isa Colon ? Tuple(1:N) : Tuple(dims)
    L=LinearIndices(T.dims)
    keys=[L[ntuple(d->d in rd ? T.dims[d]+1-c[d] : c[d],N)...] for c in (_sparse_coords(i,T.dims) for i in T.indices)]
    _like(A,_sparse_tensor_from(T.dims,keys,copy(T.values)))
end
Base.circshift(A::_OxiAll,shifts::Real)=_sparse_circshift(A,(shifts,))
Base.circshift(A::_OxiAll,shifts::Tuple{Vararg{Integer}})=_sparse_circshift(A,shifts)
function _sparse_circshift(A,shifts)
    T=_coo(A); N=ndims(T); s=ntuple(d->d<=length(shifts) ? Int(shifts[d]) : 0,N)
    L=LinearIndices(T.dims)
    keys=[L[ntuple(d->mod(c[d]-1+s[d],T.dims[d])+1,N)...] for c in (_sparse_coords(i,T.dims) for i in T.indices)]
    _like(A,_sparse_tensor_from(T.dims,keys,copy(T.values)))
end
Base.vec(A::_OxiSparse)=reshape(_coo(A),(length(A),))
Base.vec(A::SparseTensor)=reshape(A,(length(A),))
Base.reshape(A::_OxiSparse,dims::Dims)=(R=reshape(_coo(A),dims); length(dims)==2 ? _like(A,R) : R)

# ---- zero and similar keep the family -------------------------------------------------------
Base.zero(A::SparseTensor{T}) where T=SparseTensor(A.dims,Int[],T[])
Base.zero(A::ExtendedSparse)=zero(CSRMatrix(A))
# An empty sparse array of the requested size: same family for a matrix,
# SparseTensor for another order.
function Base.similar(A::_OxiSparse,::Type{S},dims::Dims{2}) where {S<:GenericFloat}
    m,n=dims
    A isa CSCMatrix && return CSCMatrix(_Shared(),m,n,ones(Int,n+1),Int[],S[])
    A isa COOMatrix && return COOMatrix(m,n,Int[],Int[],S[])
    CSRMatrix(_Shared(),m,n,ones(Int,m+1),Int[],S[])
end
Base.similar(A::_OxiSparse,::Type{S},dims::Dims{N}) where {S,N}=SparseTensor(dims,Int[],S[])
