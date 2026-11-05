# SPDX-License-Identifier: Apache-2.0
struct DIAMatrix{T} <: SparseArrays.AbstractSparseMatrix{T,Int}
    m::Int
    n::Int
    offsets::Vector{Int}
    data::Matrix{T}
    function DIAMatrix(m::Integer,n::Integer,offsets::AbstractVector{<:Integer},data::AbstractMatrix{T}) where {T<:GenericFloat}
        Base.require_one_based_indexing(offsets,data)
        min(m,n)>=0 && size(data)==(m,length(offsets)) || throw(DimensionMismatch("invalid DIA dimensions"))
        allunique(offsets) || throw(ArgumentError("duplicate DIA offset"))
        values=zeros(T,m,length(offsets))
        for (k,d) in enumerate(offsets)
            max(1,1-d)<=min(m,n-d) || throw(ArgumentError("DIA offset has no valid entries"))
            for i in max(1,1-d):min(m,n-d)
                values[i,k]=data[i,k]
            end
        end
        new{T}(m,n,Int.(offsets),values)
    end
end
Base.size(A::DIAMatrix)=(A.m,A.n)
Base.dataids(A::DIAMatrix)=Base.dataids(A.data)
function Base.getindex(A::DIAMatrix{T},i::Int,j::Int) where T
    @boundscheck checkbounds(A,i,j)
    k=findfirst(==(j-i),A.offsets); k===nothing ? zero(T) : A.data[i,k]
end
function DIAMatrix(A::CSRMatrix{T}) where T
    offsets=sort!(unique([Int(A.colind[p])-i for i in 1:A.m for p in A.rowptr[i]:A.rowptr[i+1]-1]))
    data=zeros(T,A.m,length(offsets)); mapping=Dict(d=>k for (k,d) in enumerate(offsets))
    for i in 1:A.m, p in A.rowptr[i]:A.rowptr[i+1]-1
        data[i,mapping[A.colind[p]-i]]=A.values[p]
    end
    DIAMatrix(A.m,A.n,offsets,data)
end

struct ELLMatrix{T} <: SparseArrays.AbstractSparseMatrix{T,Int}
    m::Int
    n::Int
    indices::Matrix{Int}
    values::Matrix{T}
    function ELLMatrix(m::Integer,n::Integer,indices::AbstractMatrix{<:Integer},values::AbstractMatrix{T}) where {T<:GenericFloat}
        Base.require_one_based_indexing(indices,values)
        min(m,n)>=0 && size(indices)==size(values) && size(indices,1)==m || throw(DimensionMismatch("invalid ELL dimensions"))
        jj=Matrix{Int}(indices); vv=zeros(T,size(values))
        for i in 1:m
            seen=Set{Int}()
            for k in axes(indices,2)
                j=jj[i,k]; 0<=j<=n || throw(ArgumentError("ELL column out of range"))
                iszero(j) && continue
                j in seen && throw(ArgumentError("duplicate ELL entry"))
                push!(seen,j); vv[i,k]=values[i,k]
            end
        end
        new{T}(m,n,jj,vv)
    end
end
Base.size(A::ELLMatrix)=(A.m,A.n)
Base.dataids(A::ELLMatrix)=(Base.dataids(A.indices)...,Base.dataids(A.values)...)
function Base.getindex(A::ELLMatrix{T},i::Int,j::Int) where T
    @boundscheck checkbounds(A,i,j)
    for k in axes(A.indices,2)
        A.indices[i,k]==j && return A.values[i,k]
    end
    zero(T)
end
function ELLMatrix(A::CSRMatrix{T};width=maximum(diff(A.rowptr);init=0)) where T
    width>=maximum(diff(A.rowptr);init=0) || throw(ArgumentError("ELL width cannot truncate rows"))
    jj=zeros(Int,A.m,width); vv=zeros(T,A.m,width)
    for i in 1:A.m, (k,p) in enumerate(A.rowptr[i]:A.rowptr[i+1]-1)
        jj[i,k]=A.colind[p]; vv[i,k]=A.values[p]
    end
    ELLMatrix(A.m,A.n,jj,vv)
end

struct BSRMatrix{T} <: SparseArrays.AbstractSparseMatrix{T,Int}
    m::Int
    n::Int
    blockrows::Int
    blockcols::Int
    rowptr::Vector{Int}
    colind::Vector{Int}
    blocks::Array{T,3}
    function BSRMatrix(m::Integer,n::Integer,br::Integer,bc::Integer,rowptr::AbstractVector{<:Integer},colind::AbstractVector{<:Integer},blocks::AbstractArray{T,3}) where {T<:GenericFloat}
        Base.require_one_based_indexing(rowptr,colind,blocks)
        min(m,n)>=0 && min(br,bc)>0 || throw(ArgumentError("invalid BSR dimensions"))
        size(blocks)==(br,bc,length(colind)) || throw(DimensionMismatch("BSR block storage differs"))
        # Validate the block sparsity pattern independently of block values.
        pattern=CSRMatrix(cld(m,br),cld(n,bc),Int.(rowptr),Int.(colind),ones(Float64,length(colind)))
        owned=zeros(T,size(blocks))
        for row in 1:cld(m,br), p in pattern.rowptr[row]:pattern.rowptr[row+1]-1
            col=pattern.colind[p]
            for j in 1:min(bc,n-(col-1)*bc), i in 1:min(br,m-(row-1)*br)
                owned[i,j,p]=blocks[i,j,p]
            end
        end
        new{T}(m,n,br,bc,pattern.rowptr,pattern.colind,owned)
    end
end
Base.size(A::BSRMatrix)=(A.m,A.n)
Base.dataids(A::BSRMatrix)=(Base.dataids(A.rowptr)...,Base.dataids(A.colind)...,Base.dataids(A.blocks)...)
function Base.getindex(A::BSRMatrix{T},i::Int,j::Int) where T
    @boundscheck checkbounds(A,i,j)
    row=fld(i-1,A.blockrows)+1; col=fld(j-1,A.blockcols)+1
    for p in A.rowptr[row]:A.rowptr[row+1]-1
        A.colind[p]==col && return A.blocks[mod1(i,A.blockrows),mod1(j,A.blockcols),p]
    end
    zero(T)
end
function BSRMatrix(A::CSRMatrix{T};blocksize=(4,4)) where T
    br,bc=blocksize; min(br,bc)>0 || throw(ArgumentError("positive block sizes required"))
    lists=[Dict{Int,Matrix{T}}() for _ in 1:cld(A.m,br)]
    for i in 1:A.m, p in A.rowptr[i]:A.rowptr[i+1]-1
        j=A.colind[p]; row=fld(i-1,br)+1; col=fld(j-1,bc)+1
        block=get!(()->zeros(T,br,bc),lists[row],col)
        block[mod1(i,br),mod1(j,bc)]=A.values[p]
    end
    ptr=Int[1]; columns=Int[]; blocks=Matrix{T}[]
    for row in lists
        for col in sort!(collect(keys(row)))
            push!(columns,col); push!(blocks,row[col])
        end
        push!(ptr,length(columns)+1)
    end
    data=zeros(T,br,bc,length(blocks))
    for k in eachindex(blocks)
        data[:,:,k]=blocks[k]
    end
    BSRMatrix(A.m,A.n,br,bc,ptr,columns,data)
end

struct BSCMatrix{T} <: SparseArrays.AbstractSparseMatrix{T,Int}
    transposed_blocks::BSRMatrix{T}
end
Base.size(A::BSCMatrix)=reverse(size(A.transposed_blocks))
Base.dataids(A::BSCMatrix)=Base.dataids(A.transposed_blocks)
Base.getindex(A::BSCMatrix,i::Int,j::Int)=A.transposed_blocks[j,i]
function BSCMatrix(A::CSRMatrix{T};blocksize=(4,4)) where T
    C=CSCMatrix(A); transposed=CSRMatrix(A.n,A.m,C.colptr,C.rowind,C.values)
    BSCMatrix(BSRMatrix(transposed;blocksize=reverse(blocksize)))
end

struct HYBMatrix{T} <: SparseArrays.AbstractSparseMatrix{T,Int}
    ell::ELLMatrix{T}
    coo::COOMatrix{T,Int}
    function HYBMatrix(ell::ELLMatrix{T},coo::COOMatrix{T}) where T
        size(ell)==size(coo) || throw(DimensionMismatch("HYB component dimensions differ"))
        E=ELLMatrix(ell.m,ell.n,ell.indices,ell.values)
        C=COOMatrix(coo.m,coo.n,Int.(coo.rows),Int.(coo.cols),coo.values)
        new{T}(E,C)
    end
end
Base.size(A::HYBMatrix)=size(A.ell)
Base.dataids(A::HYBMatrix)=(Base.dataids(A.ell)...,Base.dataids(A.coo)...)
Base.getindex(A::HYBMatrix,i::Int,j::Int)=A.ell[i,j]+A.coo[i,j]
function HYBMatrix(A::CSRMatrix{T};width=isempty(A.values) ? 0 : cld(length(A.values),max(1,A.m))) where T
    width>=0 || throw(ArgumentError("negative HYB width"))
    jj=zeros(Int,A.m,width); vv=zeros(T,A.m,width); rows=Int[]; cols=Int[]; values=T[]
    for i in 1:A.m, (k,p) in enumerate(A.rowptr[i]:A.rowptr[i+1]-1)
        if k<=width
            jj[i,k]=A.colind[p]; vv[i,k]=A.values[p]
        else
            push!(rows,i); push!(cols,A.colind[p]); push!(values,A.values[p])
        end
    end
    HYBMatrix(ELLMatrix(A.m,A.n,jj,vv),COOMatrix(A.m,A.n,rows,cols,values))
end

struct SELLMatrix{T} <: SparseArrays.AbstractSparseMatrix{T,Int}
    m::Int
    n::Int
    sliceheight::Int
    sliceptr::Vector{Int}
    widths::Vector{Int}
    indices::Vector{Int}
    values::Vector{T}
    permutation::Vector{Int}
    function SELLMatrix(m::Integer,n::Integer,height::Integer,ptr::AbstractVector{<:Integer},widths::AbstractVector{<:Integer},indices::AbstractVector{<:Integer},values::AbstractVector{T},permutation::AbstractVector{<:Integer}) where {T<:GenericFloat}
        Base.require_one_based_indexing(ptr,widths,indices,values,permutation)
        min(m,n)>=0 && height>=1 || throw(ArgumentError("invalid SELL dimensions"))
        slices=cld(m,height)
        length(ptr)==slices+1 && length(widths)==slices && length(indices)==length(values) || throw(DimensionMismatch("invalid SELL storage lengths"))
        ptr[1]==1 && ptr[end]==length(values)+1 && issorted(ptr) && all(>=(0),widths) || throw(ArgumentError("invalid SELL slice pointers"))
        sort(collect(permutation))==collect(1:m) || throw(ArgumentError("invalid SELL row permutation"))
        jj=zeros(Int,length(indices)); vv=zeros(T,length(values))
        for slice in 1:slices
            ptr[slice+1]-ptr[slice]==height*widths[slice] || throw(DimensionMismatch("SELL slice extent differs"))
            for lane in 1:min(height,m-(slice-1)*height)
                seen=Set{Int}()
                for k in 1:widths[slice]
                    p=ptr[slice]+(k-1)*height+lane-1; j=indices[p]
                    0<=j<=n || throw(ArgumentError("SELL column out of range"))
                    j==0 && continue
                    j in seen && throw(ArgumentError("duplicate SELL row entry"))
                    push!(seen,j); jj[p]=j; vv[p]=values[p]
                end
            end
        end
        new{T}(m,n,height,Int.(ptr),Int.(widths),jj,vv,Int.(permutation))
    end
end
Base.size(A::SELLMatrix)=(A.m,A.n)
Base.dataids(A::SELLMatrix)=(Base.dataids(A.indices)...,Base.dataids(A.values)...,Base.dataids(A.permutation)...)
function SELLMatrix(A::CSRMatrix{T};sliceheight=8,sorting_window=1) where T
    min(sliceheight,sorting_window)>0 || throw(ArgumentError("positive SELL slice/window sizes required"))
    perm=collect(1:A.m); lengths=diff(A.rowptr)
    for start in 1:sorting_window:A.m
        stop=min(A.m,start+sorting_window-1)
        perm[start:stop]=sort(perm[start:stop];by=i->lengths[i],rev=true)
    end
    ptr=Int[1]; widths=Int[]; jj=Int[]; vv=T[]
    for start in 1:sliceheight:A.m
        stop=min(A.m,start+sliceheight-1); width=maximum((lengths[perm[i]] for i in start:stop);init=0)
        push!(widths,width); base=length(jj); append!(jj,zeros(Int,sliceheight*width)); append!(vv,zeros(T,sliceheight*width))
        for slot in start:stop
            i=perm[slot]; lane=slot-start+1
            for (k,p) in enumerate(A.rowptr[i]:A.rowptr[i+1]-1)
                offset=base+(k-1)*sliceheight+lane; jj[offset]=A.colind[p]; vv[offset]=A.values[p]
            end
        end
        push!(ptr,length(jj)+1)
    end
    SELLMatrix(A.m,A.n,sliceheight,ptr,widths,jj,vv,perm)
end
function Base.getindex(A::SELLMatrix{T},i::Int,j::Int) where T
    @boundscheck checkbounds(A,i,j)
    slot=findfirst(==(i),A.permutation)::Int; slice=fld(slot-1,A.sliceheight)+1; lane=mod1(slot,A.sliceheight)
    for k in 1:A.widths[slice]
        p=A.sliceptr[slice]+(k-1)*A.sliceheight+lane-1
        A.indices[p]==j && return A.values[p]
    end
    zero(T)
end

const ExtendedSparse=Union{DIAMatrix,ELLMatrix,BSRMatrix,BSCMatrix,HYBMatrix,SELLMatrix}
function _foreach_stored(f,A::DIAMatrix)
    for (k,d) in enumerate(A.offsets), i in max(1,1-d):min(A.m,A.n-d)
        f(i,i+d,A.data[i,k])
    end
end
function _foreach_stored(f,A::ELLMatrix)
    for k in axes(A.indices,2), i in 1:A.m
        j=A.indices[i,k]; j!=0 && f(i,j,A.values[i,k])
    end
end
function _foreach_stored(f,A::BSRMatrix)
    for row in 1:cld(A.m,A.blockrows), p in A.rowptr[row]:A.rowptr[row+1]-1
        col=A.colind[p]
        for j in 1:min(A.blockcols,A.n-(col-1)*A.blockcols), i in 1:min(A.blockrows,A.m-(row-1)*A.blockrows)
            f((row-1)*A.blockrows+i,(col-1)*A.blockcols+j,A.blocks[i,j,p])
        end
    end
end
_foreach_stored(f,A::BSCMatrix)=_foreach_stored((i,j,v)->f(j,i,v),A.transposed_blocks)
function _foreach_stored(f,A::HYBMatrix)
    _foreach_stored(f,A.ell)
    for p in eachindex(A.coo.values)
        f(A.coo.rows[p],A.coo.cols[p],A.coo.values[p])
    end
end
function _foreach_stored(f,A::SELLMatrix)
    for slice in eachindex(A.widths), k in 1:A.widths[slice], lane in 1:A.sliceheight
        slot=(slice-1)*A.sliceheight+lane; slot>A.m && continue
        p=A.sliceptr[slice]+(k-1)*A.sliceheight+lane-1; j=A.indices[p]
        j!=0 && f(A.permutation[slot],j,A.values[p])
    end
end
function CSRMatrix(A::ExtendedSparse)
    T=eltype(A); rows=Int[]; cols=Int[]; values=T[]
    _foreach_stored(A) do i,j,v
        !iszero(v) && (push!(rows,i); push!(cols,j); push!(values,v))
    end
    CSRMatrix(COOMatrix(size(A,1),size(A,2),rows,cols,values))
end
CSCMatrix(A::ExtendedSparse)=CSCMatrix(CSRMatrix(A))
COOMatrix(A::ExtendedSparse)=COOMatrix(CSRMatrix(A))
for Format in (:DIAMatrix,:ELLMatrix,:BSRMatrix,:BSCMatrix,:HYBMatrix,:SELLMatrix)
    @eval $Format(A::Union{CSCMatrix,COOMatrix,ExtendedSparse};kwargs...)=$Format(CSRMatrix(A);kwargs...)
end
function sparse_mul!(y::AbstractVector{T},A::ExtendedSparse,x::AbstractVector{T},alpha0=one(T),beta0=zero(T);trans=:N,threaded=true,parallel_threshold=262144) where T
    Base.require_one_based_indexing(y,x); _check_vector_output(y); tr=_trans_kind(trans)
    size(_op(A,trans))==(length(y),length(x)) || throw(DimensionMismatch("sparse multiply dimensions differ"))
    eltype(A)===T || throw(ArgumentError("sparse element types must agree"))
    Base.mightalias(y,A) && throw(ArgumentError("output must not alias sparse storage"))
    alpha,beta=convert(T,alpha0),convert(T,beta0)
    iszero(alpha) && return _scale!(y,beta)
    threaded=threaded && generic_threadsafe(T)
    if threaded && Threads.nthreads()>1 && length(x)*Float64(length(y))>=parallel_threshold
        return sparse_mul!(y,CSRMatrix(A),x,alpha,beta;trans,threaded,parallel_threshold)
    end
    xx=Base.mightalias(x,y) ? copy(x) : x; _scale!(y,beta)
    _foreach_stored(A) do i,j,v
        if tr==0
            y[i]+=alpha*v*xx[j]
        else
            y[j]+=alpha*(tr==2 ? conj(v) : v)*xx[i]
        end
    end
    y
end
_native_mv!(y,A::ExtendedSparse,x)=sparse_mul!(y,A,x)
function sparse_mul!(Y::AbstractMatrix{T},A::ExtendedSparse,X::AbstractMatrix{T},alpha=one(T),beta=zero(T);kwargs...) where T
    Base.mightalias(Y,A) && throw(ArgumentError("output must not alias sparse storage"))
    sparse_mul!(Y,CSRMatrix(A),X,alpha,beta;kwargs...)
end
sparse_matmul(A::ExtendedSparse,B::ExtendedSparse;kwargs...)=sparse_matmul(CSRMatrix(A),CSRMatrix(B);kwargs...)
sparse_matmul(A::ExtendedSparse,B::Union{CSRMatrix,CSCMatrix,COOMatrix};kwargs...)=sparse_matmul(CSRMatrix(A),CSRMatrix(B);kwargs...)
sparse_matmul(A::Union{CSRMatrix,CSCMatrix,COOMatrix},B::ExtendedSparse;kwargs...)=sparse_matmul(CSRMatrix(A),CSRMatrix(B);kwargs...)
