# SPDX-License-Identifier: Apache-2.0
# Native local binary tile cache. No numerical external backend.
struct _OOCTileStore
    directory::String
    n::Int
    blocksize::Int
    present::BitMatrix
end
mutable struct OutOfCoreFactor{T}
    kind::Symbol
    store::_OOCTileStore
    p::Vector{Int}
    q::Vector{Int}
    panel::Vector{T}
    buffers::NTuple{4,Vector{T}}
    workspace::GemmWorkspace{T}
    payload_bytes::Int
    memory_budget::Int
    guard::ReentrantLock
    closed::Bool
end
Base.size(F::OutOfCoreFactor)=(F.store.n,F.store.n)
_ooc_tag(::Type{Float32})=UInt8(1)
_ooc_tag(::Type{Float64})=UInt8(2)
_ooc_tag(::Type{ComplexF32})=UInt8(3)
_ooc_tag(::Type{ComplexF64})=UInt8(4)
_ooc_range(S,i)=(i-1)*S.blocksize+1:min(i*S.blocksize,S.n)
_ooc_path(S,i,j)=joinpath(S.directory,"tile-$(i)-$(j).bin")
function _ooc_checksum(buffer,count)
    bytes=reinterpret(UInt8,buffer); checksum=UInt64(0xcbf29ce484222325)
    for i in 1:Base.checked_mul(count,sizeof(eltype(buffer)))
        checksum=(checksum ⊻ bytes[i])*UInt64(0x100000001b3)
    end
    checksum
end
function _ooc_write!(buffer,F,i,j)
    S=F.store; rows=length(_ooc_range(S,i)); cols=length(_ooc_range(S,j)); count=rows*cols
    count<=length(buffer) || throw(DimensionMismatch("tile write buffer too short"))
    checksum=_ooc_checksum(buffer,count)
    open(_ooc_path(S,i,j),"w") do io
        write(io,UInt64(0x4f5849424c415331)); write(io,UInt32(Base.ENDIAN_BOM)); write(io,_ooc_tag(eltype(buffer)))
        write(io,Int64(rows)); write(io,Int64(cols)); write(io,checksum)
        bytes=Base.checked_mul(count,sizeof(eltype(buffer)))
        if bytes>0
            GC.@preserve buffer Base.unsafe_write(io,Ptr{UInt8}(pointer(buffer)),UInt(bytes))
        end
    end
    S.present[i,j]=true
end
function _ooc_read!(buffer,F,i,j)
    S=F.store; rows=length(_ooc_range(S,i)); cols=length(_ooc_range(S,j)); count=rows*cols
    count<=length(buffer) || throw(DimensionMismatch("tile read buffer too short"))
    if !S.present[i,j]
        fill!(view(buffer,1:count),zero(eltype(buffer)))
    else
        path=_ooc_path(S,i,j); bytes=Base.checked_mul(count,sizeof(eltype(buffer)))
        filesize(path)==37+bytes || throw(ArgumentError("invalid tile file size"))
        open(path,"r") do io
            read(io,UInt64)==UInt64(0x4f5849424c415331) || throw(ArgumentError("invalid tile magic"))
            read(io,UInt32)==UInt32(Base.ENDIAN_BOM) || throw(ArgumentError("tile byte order differs"))
            read(io,UInt8)==_ooc_tag(eltype(buffer)) || throw(ArgumentError("tile scalar type differs"))
            read(io,Int64)==rows && read(io,Int64)==cols || throw(ArgumentError("tile shape differs"))
            checksum=read(io,UInt64)
            if bytes>0
                GC.@preserve buffer Base.unsafe_read(io,Ptr{UInt8}(pointer(buffer)),UInt(bytes))
            end
            _ooc_checksum(buffer,count)==checksum || throw(ArgumentError("tile checksum mismatch"))
        end
    end
    reshape(view(buffer,1:count),rows,cols)
end
function _ooc_payload(::Type{T},n,b,workers) where T
    config=GemmConfig(T); depth=min(b,config.kc)
    a=(min(b,config.mc)+15)*depth+cld(128,sizeof(T))
    c=(min(b,config.nc)+15)*depth+cld(128,sizeof(T))
    scalars=Base.checked_add(Base.checked_mul(n,b),Base.checked_mul(4,Base.checked_mul(b,b)))
    scalars=Base.checked_add(scalars,Base.checked_mul(workers,Base.checked_add(a,c)))
    Base.checked_mul(scalars,sizeof(T))
end
function _ooc_create(::Type{T},n,kind,q,blocksize,budget,directory,threaded) where {T<:NativeFloat}
    blocksize>=1 && budget>=0 || throw(ArgumentError("positive tile size and nonnegative byte budget required"))
    workers=threaded ? Threads.nthreads() : 1; tile=max(1,min(Int(blocksize),max(1,n)))
    while _ooc_payload(T,n,tile,workers)>budget && tile>1
        tile=max(1,tile÷2)
    end
    b=tile   # final tile size, assigned once: captured by the ntuple closure below
    _ooc_payload(T,n,b,workers)<=budget || throw(ArgumentError("memory budget cannot hold one column panel and tile workspace"))
    panel=Vector{T}(undef,Base.checked_mul(n,b)); buffers=ntuple(_->Vector{T}(undef,b*b),4)
    workspace=GemmWorkspace(T,b,b,b;workers)
    payload=sizeof(T)*(length(panel)+sum(length,buffers)+sum(length(B.a)+length(B.b) for B in workspace.buffers))
    payload<=budget || error("out-of-core workspace accounting mismatch")
    path=mktempdir(directory;prefix="oxiblas-tiles-")
    F=OutOfCoreFactor(kind,_OOCTileStore(path,n,b,falses(cld(n,b),cld(n,b))),kind===:cholesky ? copy(q) : collect(1:n),
                      copy(q),panel,buffers,workspace,payload,Int(budget),ReentrantLock(),false)
    finalizer(F) do factor
        try
            close(factor)
        catch
            # Explicit close is the supported error-reporting cleanup path.
        end
    end
    F
end
function Base.close(F::OutOfCoreFactor)
    lock(F.guard) do
        F.closed && return nothing
        rm(F.store.directory;recursive=true,force=true); F.closed=true
        nothing
    end
end
function out_of_core_memory_usage(F::OutOfCoreFactor)
    (;payload_bytes=F.payload_bytes,budget_bytes=F.memory_budget,
      metadata_bytes=max(0,Base.summarysize(F)-F.payload_bytes),
      includes_input_output=false,includes_runtime_overhead=false)
end
function _ooc_initialize!(F,C)
    S=F.store; n=S.n; iq=invperm(F.q); tiles=size(S.present,1); scratch=F.buffers[1]
    for i in 1:tiles
        range=_ooc_range(S,i); rows=length(range)
        panel=reshape(view(F.panel,1:rows*n),rows,n); fill!(panel,zero(eltype(F.panel)))
        for (localrow,position) in enumerate(range)
            original=F.kind===:cholesky ? F.q[position] : position
            for e in C.rowptr[original]:C.rowptr[original+1]-1
                panel[localrow,iq[C.colind[e]]]=C.values[e]
            end
        end
        for j in 1:(F.kind===:cholesky ? i : tiles)
            columns=_ooc_range(S,j); count=rows*length(columns)
            tile=reshape(view(scratch,1:count),rows,length(columns)); copyto!(tile,view(panel,:,columns))
            any(!iszero,tile) && _ooc_write!(scratch,F,i,j)
        end
    end
end
function _ooc_swap_rows!(F,a,b,except_column)
    a==b && return
    S=F.store; ia=cld(a,S.blocksize); ib=cld(b,S.blocksize)
    ra=a-first(_ooc_range(S,ia))+1; rb=b-first(_ooc_range(S,ib))+1
    for j in 1:size(S.present,2)
        j==except_column && continue
        A=_ooc_read!(F.buffers[1],F,ia,j)
        if ia==ib
            for col in axes(A,2)
                A[ra,col],A[rb,col]=A[rb,col],A[ra,col]
            end
            _ooc_write!(F.buffers[1],F,ia,j)
        else
            B=_ooc_read!(F.buffers[2],F,ib,j)
            for col in axes(A,2)
                A[ra,col],B[rb,col]=B[rb,col],A[ra,col]
            end
            _ooc_write!(F.buffers[1],F,ia,j); _ooc_write!(F.buffers[2],F,ib,j)
        end
    end
end
function _ooc_lu_factorize!(F;threaded=false)
    S=F.store; n=S.n; tiles=size(S.present,1)
    for k in 1:tiles
        columns=_ooc_range(S,k); start=first(columns); width=length(columns); rows=n-start+1
        panel=reshape(view(F.panel,1:rows*width),rows,width)
        for i in k:tiles
            tile=_ooc_read!(F.buffers[1],F,i,k); localrows=(first(_ooc_range(S,i))-start+1):(last(_ooc_range(S,i))-start+1)
            copyto!(view(panel,localrows,:),tile)
        end
        for j in 1:width
            pivot=j; largest=abs(panel[j,j])
            for i in j+1:rows
                abs(panel[i,j])>largest && (largest=abs(panel[i,j]); pivot=i)
            end
            isfinite(largest) && largest>0 || throw(SingularException(start+j-1))
            if pivot!=j
                for col in 1:width
                    panel[j,col],panel[pivot,col]=panel[pivot,col],panel[j,col]
                end
                a,b=start+j-1,start+pivot-1
                _ooc_swap_rows!(F,a,b,k); F.p[a],F.p[b]=F.p[b],F.p[a]
            end
            diagonal=panel[j,j]
            for i in j+1:rows
                panel[i,j]/=diagonal
                for col in j+1:width
                    panel[i,col]=_native_madd(-panel[i,j],panel[j,col],panel[i,col])
                end
            end
        end
        all(isfinite,panel) || throw(ArgumentError("out-of-core LU panel overflow"))
        for i in k:tiles
            range=_ooc_range(S,i); localrows=(first(range)-start+1):(last(range)-start+1); count=length(range)*width
            tile=reshape(view(F.buffers[1],1:count),length(range),width); copyto!(tile,view(panel,localrows,:))
            _ooc_write!(F.buffers[1],F,i,k)
        end
        for j in k+1:tiles
            upper=_ooc_read!(F.buffers[3],F,k,j)
            # A full tile leaf avoids recursive B-view alias snapshots outside
            # the numeric byte budget. GEMM updates reuse the packing workspace.
            trsm!(upper,view(panel,1:width,:);uplo=:L,unit=true,blocksize=width,threaded,workspace=F.workspace)
            all(isfinite,upper) || throw(ArgumentError("out-of-core LU upper panel overflow"))
            _ooc_write!(F.buffers[3],F,k,j)
            for i in k+1:tiles
                localrows=(first(_ooc_range(S,i))-start+1):(last(_ooc_range(S,i))-start+1); lower=view(panel,localrows,:)
                trailing=_ooc_read!(F.buffers[4],F,i,j)
                gemm!(trailing,lower,upper,-one(eltype(F.panel)),one(eltype(F.panel));threaded,workspace=F.workspace)
                all(isfinite,trailing) || throw(ArgumentError("out-of-core LU update overflow"))
                _ooc_write!(F.buffers[4],F,i,j)
            end
        end
    end
end
function _ooc_cholesky_factorize!(F;threaded=false)
    S=F.store; T=eltype(F.panel); tiles=size(S.present,1)
    for k in 1:tiles
        diagonal=_ooc_read!(F.buffers[1],F,k,k); width=size(diagonal,1)
        for j in 1:width
            d=real(diagonal[j,j])
            for col in 1:j-1
                d-=abs2(diagonal[j,col])
            end
            isfinite(d) && d>0 || throw(PosDefException(first(_ooc_range(S,k))+j-1))
            diagonal[j,j]=sqrt(d)
            for i in j+1:width
                value=diagonal[i,j]
                for col in 1:j-1
                    value=_native_madd(-diagonal[i,col],conj(diagonal[j,col]),value)
                end
                diagonal[i,j]=value/diagonal[j,j]
            end
        end
        all(isfinite,diagonal) || throw(ArgumentError("out-of-core Cholesky diagonal overflow"))
        _ooc_write!(F.buffers[1],F,k,k)
        for i in k+1:tiles
            panel=_ooc_read!(F.buffers[2],F,i,k)
            trsm!(panel,diagonal;side=:R,uplo=:L,trans=:C,blocksize=width,threaded,workspace=F.workspace)
            all(isfinite,panel) || throw(ArgumentError("out-of-core Cholesky panel overflow"))
            _ooc_write!(F.buffers[2],F,i,k)
        end
        for j in k+1:tiles
            right=_ooc_read!(F.buffers[2],F,j,k)
            for i in j:tiles
                left=_ooc_read!(F.buffers[3],F,i,k); trailing=_ooc_read!(F.buffers[4],F,i,j)
                gemm!(trailing,left,right,-one(T),one(T);transB=:C,threaded,workspace=F.workspace)
                all(isfinite,trailing) || throw(ArgumentError("out-of-core Cholesky update overflow"))
                _ooc_write!(F.buffers[4],F,i,j)
            end
        end
    end
end
function _out_of_core(A,kind;blocksize::Integer=128,memory_budget::Integer=64*1024^2,
                      directory=tempdir(),ordering=:natural,uplo=:L,threaded=false)
    n=_square(A); T=eltype(A)
    T<:NativeFloat || throw(ArgumentError("disk factors require native floating point scalars"))
    C=kind===:cholesky ? _sparse_hermitian(A,uplo) : CSRMatrix(COOMatrix(CSRMatrix(A)))
    all(isfinite,C.values) || throw(ArgumentError("finite disk factorization input required"))
    q=sparse_ordering(C,ordering)
    F=_ooc_create(T,n,kind,q,blocksize,memory_budget,directory,threaded)
    try
        _ooc_initialize!(F,C)
        kind===:lu ? _ooc_lu_factorize!(F;threaded) : _ooc_cholesky_factorize!(F;threaded)
        F
    catch
        close(F); rethrow()
    end
end
"""Native tiled disk LU with global panel row pivoting. The byte budget covers
numeric panel/tile/packing payload, not CSR input, returned solutions, metadata
or process RSS. Tile size shrinks to fit; an insufficient budget is rejected.
Call close on the returned owned cache when finished; GC cleanup is best effort.
"""
out_of_core_lu(A::NativeSparse;kwargs...)=_out_of_core(A,:lu;kwargs...)
"""Native tiled disk Hermitian Cholesky, with the same cache/budget contract as
out_of_core_lu. Only uplo is read. Cache files are local native-endian tiles,
checksummed and shape/type checked; they are not portable persistent checkpoints.
"""
out_of_core_cholesky(A::NativeSparse;kwargs...)=_out_of_core(A,:cholesky;kwargs...)

# Triangular sweep reads one tile at a time; transposition never materializes
# another factor or a dense global matrix.
function _ooc_triangular!(work,F,lower,unit,tr)
    S=F.store; tiles=size(S.present,1); effective_lower=tr==0 ? lower : !lower
    visit=effective_lower ? (1:tiles) : (tiles:-1:1)
    for i in visit
        rows=_ooc_range(S,i)
        preceding=effective_lower ? (1:i-1) : (i+1:tiles)
        for j in preceding
            columns=_ooc_range(S,j)
            tile=tr==0 ? _ooc_read!(F.buffers[1],F,i,j) : _ooc_read!(F.buffers[1],F,j,i)
            for c in eachindex(columns), r in eachindex(rows)
                value=tr==0 ? tile[r,c] : tr==1 ? tile[c,r] : conj(tile[c,r])
                work[rows[r]]=_native_madd(-value,work[columns[c]],work[rows[r]])
            end
        end
        diagonal=_ooc_read!(F.buffers[1],F,i,i)
        localorder=effective_lower ? (1:length(rows)) : (length(rows):-1:1)
        for r in localorder
            preceding_local=effective_lower ? (1:r-1) : (r+1:length(rows))
            for c in preceding_local
                value=tr==0 ? diagonal[r,c] : tr==1 ? diagonal[c,r] : conj(diagonal[c,r])
                work[rows[r]]=_native_madd(-value,work[rows[c]],work[rows[r]])
            end
            if !unit
                d=tr==2 ? conj(diagonal[r,r]) : diagonal[r,r]
                work[rows[r]]/=d
            end
        end
    end
    work
end
function solve(F::OutOfCoreFactor{T},b::AbstractVecOrMat{T};trans=:N) where T
    Base.require_one_based_indexing(b); n=F.store.n; tr=_trans_kind(trans)
    size(b,1)==n || throw(DimensionMismatch("disk solve RHS differs"))
    lock(F.guard) do
        F.closed && throw(ArgumentError("out-of-core factor is closed"))
        (Base.mightalias(b,F.panel) || any(B->Base.mightalias(b,B),F.buffers)) && throw(ArgumentError("RHS aliases disk workspace"))
        rhs=ndims(b)==1 ? reshape(b,n,1) : b; X=Matrix{T}(undef,size(rhs)); work=view(F.panel,1:n)
        chol=F.kind===:cholesky; conjugated=chol && tr==1 && T<:Complex
        input=chol || tr==0 ? F.p : F.q; output=chol || tr==0 ? F.q : F.p
        for col in axes(rhs,2)
            for i in 1:n
                value=rhs[input[i],col]; work[i]=conjugated ? conj(value) : value
            end
            if chol
                _ooc_triangular!(work,F,true,false,0); _ooc_triangular!(work,F,true,false,2)
            elseif tr==0
                _ooc_triangular!(work,F,true,true,0); _ooc_triangular!(work,F,false,false,0)
            else
                _ooc_triangular!(work,F,false,false,tr); _ooc_triangular!(work,F,true,true,tr)
            end
            for i in 1:n
                X[output[i],col]=conjugated ? conj(work[i]) : work[i]
            end
        end
        ndims(b)==1 ? vec(X) : X
    end
end
