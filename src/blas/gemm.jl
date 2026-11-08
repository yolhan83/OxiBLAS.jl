# SPDX-License-Identifier: Apache-2.0

"""
    GemmConfig(::Type{T}=Float64; mc=576, kc=..., nc=2048, parallel_threshold=8_000_000, small_threshold=8)

Cache blocking. Defaults adapt upstream MC=576, KC=448/896, NC≈2048.
"""
struct GemmConfig
    mc::Int
    kc::Int
    nc::Int
    parallel_threshold::Int
    small_threshold::Int
    function GemmConfig(::Type{T}=Float64; mc=576, kc=T<:NativeFloat ? (sizeof(T)<8 ? 896 : 448) : 64,
                        nc=2048, parallel_threshold=8_000_000, small_threshold=8) where T
        isconcretetype(T) || throw(ArgumentError("a concrete scalar type is required"))
        min(mc,kc,nc) > 0 || throw(ArgumentError("block sizes must be positive"))
        min(parallel_threshold,small_threshold) >= 0 || throw(ArgumentError("thresholds must be nonnegative"))
        new(mc,kc,nc,parallel_threshold,small_threshold)
    end
end

struct _PackBuffer{T}
    a::Vector{T}
    b::Vector{T}
    ai::Int
    bi::Int
end

function _aligned_vector(::Type{T}, n::Int) where {T<:NativeFloat}
    extra = cld(128,sizeof(T))
    v = Vector{T}(undef, Base.checked_add(n,extra))
    offset = Int(mod(-UInt(pointer(v)), UInt(128))) ÷ sizeof(T) + 1
    v, offset
end

"""
    GemmWorkspace(T, m, n, k; config=GemmConfig(T), workers=Threads.nthreads())

Reusable panels padded toward 128-byte alignment, one pair per worker.
SIMD loads require only natural real-scalar alignment, never user-array alignment. A workspace serializes
concurrent calls that share it; separate workspaces run independently. Task
migration never changes buffer ownership. Reconstruct if dimensions/config grow.
"""
mutable struct GemmWorkspace{T}
    buffers::Vector{_PackBuffer{T}}
    acap::Int
    bcap::Int
    guard::ReentrantLock
    children::Vector{GemmWorkspace{T}}
    tiles::Vector{Matrix{T}}
    jobs::Vector{Tuple{Int,Int}}
    jobkey::Tuple{Int,Int,Bool}
    edge_tiles::Vector{Matrix{T}}
end

GemmWorkspace(buffers::Vector{_PackBuffer{T}},acap::Int,bcap::Int,guard::ReentrantLock) where T =
    GemmWorkspace{T}(buffers,acap,bcap,guard,GemmWorkspace{T}[],Matrix{T}[],Tuple{Int,Int}[],(0,0,false),Matrix{T}[])

# Caller owns the parent's guard for the entire lifetime of these borrowed
# contexts. Child locks are distinct: spawned tasks never re-enter that guard.
function _workspace_children!(w::GemmWorkspace,workers)
    while length(w.children)<workers
        i=length(w.children)+1
        push!(w.children,GemmWorkspace([w.buffers[i]],w.acap,w.bcap,ReentrantLock()))
    end
    w.children
end

function _workspace_tiles!(w::GemmWorkspace{T},workers,bs) where T
    while length(w.tiles)<workers
        push!(w.tiles,Matrix{T}(undef,bs,bs))
    end
    for i in 1:workers
        size(w.tiles[i],1)<bs && (w.tiles[i]=Matrix{T}(undef,bs,bs))
    end
    w.tiles
end

function _workspace_edge_tiles!(w::GemmWorkspace{T},workers,mr,nr) where T
    while length(w.edge_tiles)<workers
        push!(w.edge_tiles,Matrix{T}(undef,mr,nr))
    end
    for i in 1:workers
        tile=w.edge_tiles[i]
        size(tile)!=(mr,nr) && (w.edge_tiles[i]=Matrix{T}(undef,mr,nr))
    end
    w.edge_tiles
end

function GemmWorkspace(::Type{T}, m::Integer, n::Integer, k::Integer;
                       config=GemmConfig(T), workers=Threads.nthreads()) where {T<:NativeFloat}
    isconcretetype(T) || throw(ArgumentError("a concrete native scalar type is required"))
    min(m,n,k) >= 0 || throw(ArgumentError("negative dimension"))
    workers > 0 || throw(ArgumentError("workers must be positive"))
    # Covers every supported micro-panel shape, including padding.
    acap = Base.checked_mul(Base.checked_add(min(m,config.mc),15),min(k,config.kc))
    bcap = Base.checked_mul(Base.checked_add(min(n,config.nc),15),min(k,config.kc))
    buffers = map(1:workers) do _
        a, ai = _aligned_vector(T, acap)
        b, bi = _aligned_vector(T, bcap)
        _PackBuffer(a,b,ai,bi)
    end
    GemmWorkspace(buffers,acap,bcap,ReentrantLock())
end

# Panels for calls without a supplied workspace are taken from a small pool
# and returned afterwards, as OpenBLAS keeps per-thread buffers: allocating
# them per call (megabytes per worker) made 512x512 products on 18 threads
# run at 5x their minimum time (GC, page faults). A call takes a free entry
# that fits or allocates one; at most _WORKSPACE_POOL_MAX entries per type
# are kept (bounded by mc*kc and nc*kc panels per worker).
const _WORKSPACE_POOL=(GemmWorkspace{Float32}[],GemmWorkspace{Float64}[],GemmWorkspace{ComplexF32}[],GemmWorkspace{ComplexF64}[])
const _WORKSPACE_POOL_LOCK=ReentrantLock()
const _WORKSPACE_POOL_MAX=4
_workspace_pool(::Type{Float32})=_WORKSPACE_POOL[1]
_workspace_pool(::Type{Float64})=_WORKSPACE_POOL[2]
_workspace_pool(::Type{ComplexF32})=_WORKSPACE_POOL[3]
_workspace_pool(::Type{ComplexF64})=_WORKSPACE_POOL[4]
function _pooled_workspace(f::F,::Type{T},workspace,m,n,k,config,workers) where {F,T<:NativeFloat}
    workspace===nothing || return f(workspace)
    acap=(min(m,config.mc)+15)*min(k,config.kc); bcap=(min(n,config.nc)+15)*min(k,config.kc)
    pool=_workspace_pool(T)
    # The pooled entry is the lock block's value: w is assigned once, so the
    # closure in `finally` captures it without a Core.Box.
    found=lock(_WORKSPACE_POOL_LOCK) do
        # Smallest entry that fits, so that large ones stay for large calls.
        best=0
        for i in eachindex(pool)
            c=pool[i]
            if length(c.buffers)>=workers && c.acap>=acap && c.bcap>=bcap && (best==0 || _workspace_size(c)<_workspace_size(pool[best]))
                best=i
            end
        end
        best>0 ? popat!(pool,best) : nothing
    end
    w=found===nothing ? GemmWorkspace(T,m,n,k;config,workers) : found
    try
        f(w)
    finally
        lock(_WORKSPACE_POOL_LOCK) do
            if length(pool)<_WORKSPACE_POOL_MAX
                push!(pool,w)
            else
                # A full pool keeps the largest entries: dropping the large
                # ones reallocated them at every call when a factorization
                # mixed panel sizes (61 MB and 46% GC per 1024 Cholesky).
                i=argmin(_workspace_size.(pool))
                _workspace_size(pool[i])<_workspace_size(w) && (pool[i]=w)
            end
        end
    end
end
_workspace_size(w::GemmWorkspace)=length(w.buffers)*(w.acap+w.bcap)

@inline function _trans_kind(trans)
    trans in (:N,'N',:n,'n') && return 0
    trans in (:T,'T',:t,'t') && return 1
    trans in (:C,'C',:c,'c') && return 2
    throw(ArgumentError("transpose must be N, T or C"))
end
@inline function _op(A,trans)
    kind=_trans_kind(trans)
    kind==0 ? A : kind==1 ? transpose(A) : adjoint(A)
end

function _gemm_dims(C,A,B)
    Base.require_one_based_indexing(C,A,B)
    size(A,2) == size(B,1) && size(C) == (size(A,1),size(B,2)) ||
        throw(DimensionMismatch("GEMM: C$(size(C)), A$(size(A)), B$(size(B))"))
    _check_output_layout(C)
    (size(C,1),size(C,2),size(A,2))
end

_check_output_layout(C::AbstractVector)=_check_vector_output(C)
function _check_output_layout(C::AbstractMatrix)
    m,n=size(C)
    (m==0 || n==0) && return nothing
    if C isa StridedMatrix
        s1,s2=abs(stride(C,1)),abs(stride(C,2))
        (m>1 && s1==0 || n>1 && s2==0) && throw(ArgumentError("output entries overlap"))
        if m>1 && n>1
            g=gcd(s1,s2)
            m>s2÷g && n>s1÷g && throw(ArgumentError("output entries overlap"))
        end
    elseif C isa SubArray
        # Recursion over the index tuple, not a loop: the tuple is
        # heterogeneous (ranges and integers) and a loop variable over it
        # was boxed (one allocation per Level 1 call on a view).
        _views_unique(parentindices(C)) || throw(ArgumentError("output view contains repeated indices"))
    elseif C isa Union{Transpose,Adjoint}
        _check_output_layout(parent(C))
    elseif C isa Union{Base.ReshapedArray,Base.PermutedDimsArray}
        _check_nd_output(C)
    end
    nothing
end

# Base.mightalias does not look through Transpose/Adjoint: a view of an
# adjoint view was always taken to overlap any other view of the same
# matrix, so GEMMs inside recursive TRSM/Cholesky copied an operand at every
# call (9 MB per 1024 Cholesky). For compositions of contiguous-range views
# and transposes of one Matrix, _block gives the rectangle of that matrix an
# operand covers (rows, columns, whether its own axes are swapped); disjoint
# rectangles cannot alias. Anything else falls back to Base.mightalias.
_block(A::Matrix)=(A,axes(A,1),axes(A,2),false)
function _block(A::Union{Transpose,Adjoint})
    b=_block(parent(A)); b===nothing && return nothing
    (b[1],b[2],b[3],!b[4])
end
function _block(A::SubArray{<:Any,2,<:Any,<:Tuple{AbstractUnitRange,AbstractUnitRange}})
    b=_block(parent(A)); b===nothing && return nothing
    base,r,c,flip=b; I,J=parentindices(A)
    flip ? (base,r[J],c[I],true) : (base,r[I],c[J],false)
end
# Column segment A[I, j] and row segment A[i, J] of a matrix (Level 1 kernels
# on two columns of one matrix, as in Householder QR updates).
function _block(A::SubArray{<:Any,1,<:Any,<:Tuple{AbstractUnitRange,Integer}})
    b=_block(parent(A)); b===nothing && return nothing
    base,r,c,flip=b; I,j=parentindices(A)
    flip ? (base,r[j:j],c[I],true) : (base,r[I],c[j:j],false)
end
function _block(A::SubArray{<:Any,1,<:Any,<:Tuple{Integer,AbstractUnitRange}})
    b=_block(parent(A)); b===nothing && return nothing
    base,r,c,flip=b; i,J=parentindices(A)
    flip ? (base,r[J],c[i:i],true) : (base,r[i:i],c[J],false)
end
_block(A)=nothing
function _mightalias(A,B)
    a=_block(A); b=_block(B)
    (a===nothing || b===nothing) && return Base.mightalias(A,B)
    a[1]===b[1] || return false
    !isempty(intersect(a[2],b[2])) && !isempty(intersect(a[3],b[3]))
end

function _scale!(C, β)
    isone(β) && return C
    if iszero(β)
        fill!(C,zero(eltype(C)))
    else
        @inbounds for i in eachindex(C)
            C[i] *= β
        end
    end
    C
end

@inline _combine(v,c,α,β) = iszero(β) ? α*v : α*v+β*c

@inline _native_madd(a::T,b::T,c::T) where {T<:RealFloat}=muladd(a,b,c)
@inline _native_madd(a::T,b::T,c::T) where {T<:AbstractFloat}=a*b+c
@inline function _native_madd(a::T,b::T,c::T) where {T<:Complex}
    real(T)<:RealFloat || return a*b+c
    ar,ai=real(a),imag(a); br,bi=real(b),imag(b)
    complex(muladd(ar,br,muladd(-ai,bi,real(c))),muladd(ar,bi,muladd(ai,br,imag(c))))
end

# Packing-free four-row register strips for small/skinny products. Neither
# dimensions nor K enter the type domain, so compile cost stays bounded.
function _small_gemm!(C,A,B,α,β)
    m,n=size(C); k=size(A,2); T=eltype(C); stop=m-mod(m,4)
    @inbounds for j in 1:n
        for row in 1:4:stop
            a=b=c=d=zero(T)
            for p in 1:k
                z=B[p,j]
                a=_native_madd(A[row,p],z,a); b=_native_madd(A[row+1,p],z,b)
                c=_native_madd(A[row+2,p],z,c); d=_native_madd(A[row+3,p],z,d)
            end
            C[row,j]=iszero(β) ? α*a : α*a+β*C[row,j]
            C[row+1,j]=iszero(β) ? α*b : α*b+β*C[row+1,j]
            C[row+2,j]=iszero(β) ? α*c : α*c+β*C[row+2,j]
            C[row+3,j]=iszero(β) ? α*d : α*d+β*C[row+3,j]
        end
        for row in stop+1:m
            s=zero(T)
            for p in 1:k
                s=_native_madd(A[row,p],B[p,j],s)
            end
            C[row,j]=iszero(β) ? α*s : α*s+β*C[row,j]
        end
    end
    C
end

function _reference!(C,A,B,α,β)
    m,n = size(C)
    k = size(A,2)
    @inbounds for j in 1:n, i in 1:m
        s = zero(eltype(C))
        for p in 1:k
            s = muladd(A[i,p],B[p,j],s)
        end
        # Keep the branch outside the load expression: β=0 must not read C.
        C[i,j] = iszero(β) ? α*s : α*s+β*C[i,j]
    end
    C
end

function _blocked!(C,A,B,α,β,config)
    _scale!(C,β)
    m,n,k = size(A,1),size(B,2),size(A,2)
    @inbounds for jc in 1:config.nc:n, pc in 1:config.kc:k, ic in 1:config.mc:m
        for j in jc:min(n,jc+config.nc-1), p in pc:min(k,pc+config.kc-1)
            b = α*B[p,j]
            @simd for i in ic:min(m,ic+config.mc-1)
                C[i,j] = muladd(A[i,p],b,C[i,j])
            end
        end
    end
    C
end

# A: [micro-panel][p][i]; B: [micro-panel][p][j]. Tails zero padded.
# Index through Julia's array interface: transpose, adjoint, negative and
# nonunit strides are all respected. Unsafe writes address owned buffers only.
function _pack_a!(dst::Ptr{T}, A, ic, pc, ib, kb, mr) where {T<:NativeFloat}
    @inbounds for ii in 0:mr:ib-1, p in 0:kb-1, i in 0:mr-1
        v = ii+i < ib ? A[ic+ii+i,pc+p] : zero(T)
        unsafe_store!(dst,v,ii*kb+p*mr+i+1)
    end
end

function _pack_b!(dst::Ptr{T}, B, pc, jc, kb, jb, nr) where {T<:NativeFloat}
    @inbounds for jj in 0:nr:jb-1, p in 0:kb-1, j in 0:nr-1
        v = jj+j < jb ? B[pc+p,jc+jj+j] : zero(T)
        unsafe_store!(dst,v,jj*kb+p*nr+j+1)
    end
end

function _scalar_micro!(C,ic,jc,a::Ptr{T},b::Ptr{T},k,α,β,mr,nr,ib,jb) where {T<:NativeFloat}
    @inbounds for j in 0:jb-1, i in 0:ib-1
        s=zero(T)
        for p in 0:k-1
            s=muladd(unsafe_load(a,p*mr+i+1),unsafe_load(b,p*nr+j+1),s)
        end
        C[ic+i,jc+j] = iszero(β) ? α*s : α*s+β*C[ic+i,jc+j]
    end
    nothing
end

@generated function _register_micro!(C,ic,jc,a::Ptr{T},b::Ptr{T},k,α,β,::Val{M},::Val{N}) where {T<:NativeFloat,M,N}
    init,step,stores=Expr(:block),Expr(:block),Expr(:block)
    for j in 0:N-1, i in 0:M-1
        s=Symbol(:s_,i,:_,j)
        push!(init.args,:($s=zero(T)))
        push!(step.args,:($s=muladd(unsafe_load(a,p*$M+$i+1),unsafe_load(b,p*$N+$j+1),$s)))
        push!(stores.args,:(C[ic+$i,jc+$j]=iszero(β) ? α*$s : α*$s+β*C[ic+$i,jc+$j]))
    end
    quote
        $init
        for p in 0:k-1
            $step
        end
        @inbounds $stores
        nothing
    end
end

@inline _can_pointer(C) = false
@inline _can_pointer(C::StridedMatrix) = stride(C,1)==1 && stride(C,2)>=max(1,size(C,1))

# tri restricts the update to C's upper (1) or lower (2) triangle, as in
# SYRK/HERK/GEMMT: micro tiles outside it are skipped, and tiles crossing the
@inline function _call_simd_micro!(c::Ptr{T}, ldc::Int, a::Ptr{T}, b::Ptr{T}, k::Int, α::T, β::T, w::Int, r::Int, nr::Int) where {T<:RealFloat}
    if w == 2 && r == 4 && nr == 6
        _simd_micro!(c, ldc, a, b, k, α, β, Val(2), Val(4), Val(6))
    elseif w == 4 && r == 2 && nr == 8
        _simd_micro!(c, ldc, a, b, k, α, β, Val(4), Val(2), Val(8))
    elseif w == 4 && r == 2 && nr == 6
        _simd_micro!(c, ldc, a, b, k, α, β, Val(4), Val(2), Val(6))
    elseif w == 8 && r == 2 && nr == 6
        _simd_micro!(c, ldc, a, b, k, α, β, Val(8), Val(2), Val(6))
    elseif w == 8 && r == 1 && nr == 8
        _simd_micro!(c, ldc, a, b, k, α, β, Val(8), Val(1), Val(8))
    elseif w == 16 && r == 1 && nr == 16
        _simd_micro!(c, ldc, a, b, k, α, β, Val(16), Val(1), Val(16))
    else
        _simd_micro!(c, ldc, a, b, k, α, β, Val(1), Val(4), Val(4))
    end
end

@inline function _call_simd_complex_micro!(c::Ptr{T}, ldc::Int, a::Ptr{T}, b::Ptr{T}, k::Int, α::T, β::T, w::Int, r::Int, nr::Int) where {T<:Complex}
    if w == 1 && r == 4 && nr == 3
        _simd_complex_micro!(c, ldc, a, b, k, α, β, Val(1), Val(4), Val(3))
    elseif w == 2 && r == 2 && nr == 4
        _simd_complex_micro!(c, ldc, a, b, k, α, β, Val(2), Val(2), Val(4))
    elseif w == 2 && r == 2 && nr == 3
        _simd_complex_micro!(c, ldc, a, b, k, α, β, Val(2), Val(2), Val(3))
    elseif w == 4 && r == 2 && nr == 6
        _simd_complex_micro!(c, ldc, a, b, k, α, β, Val(4), Val(2), Val(6))
    elseif w == 4 && r == 1 && nr == 4
        _simd_complex_micro!(c, ldc, a, b, k, α, β, Val(4), Val(1), Val(4))
    elseif w == 8 && r == 1 && nr == 8
        _simd_complex_micro!(c, ldc, a, b, k, α, β, Val(8), Val(1), Val(8))
    else
        _simd_complex_micro!(c, ldc, a, b, k, α, β, Val(1), Val(4), Val(4))
    end
end

@inline function _call_register_micro!(C::AbstractMatrix{T}, ic, jc, a::Ptr{T}, b::Ptr{T}, k, α, β, mr::Int, nr::Int) where T
    if mr == 8 && nr == 6
        _register_micro!(C, ic, jc, a, b, k, α, β, Val(8), Val(6))
    elseif mr == 8 && nr == 8
        _register_micro!(C, ic, jc, a, b, k, α, β, Val(8), Val(8))
    elseif mr == 4 && nr == 3
        _register_micro!(C, ic, jc, a, b, k, α, β, Val(4), Val(3))
    else
        _register_micro!(C, ic, jc, a, b, k, α, β, Val(4), Val(4))
    end
end

# diagonal go through the edge scratch tile (tri requires algorithm :simd).
function _packed_region!(C::AbstractMatrix{T},A,B,α,β,buf,config,rows,cols,
                         W::Int,R::Int,N::Int,algorithm,edge=nothing,tri=0) where T
    mr,nr=W*R,N
    k=size(A,2)
    # Edge tiles: panels are zero padded, so the full SIMD tile is computed
    # into this scratch tile and only its valid part is added to C (the
    # scalar edge kernel cost about 15% at 256x256 Float64, where 4 of 256
    # columns fall in a partial 6-column tile).
    if edge===nothing && algorithm===:simd && _can_pointer(C) && (T<:Complex || (T<:RealFloat && W>1))
        edge=Matrix{T}(undef,mr,nr)
    end
    GC.@preserve buf C edge begin
        pa=pointer(buf.a,buf.ai)
        pb=pointer(buf.b,buf.bi)
        for jc in first(cols):config.nc:last(cols)
            jb=min(config.nc,last(cols)-jc+1)
            for pc in 1:config.kc:k
                kb=min(config.kc,k-pc+1)
                _pack_b!(pb,B,pc,jc,kb,jb,nr)
                bp=pc==1 ? β : one(T)
                for ic in first(rows):config.mc:last(rows)
                    ib=min(config.mc,last(rows)-ic+1)
                    (tri==1 && ic>jc+jb-1 || tri==2 && ic+ib-1<jc) && continue
                    _pack_a!(pa,A,ic,pc,ib,kb,mr)
                    for jj in 0:nr:jb-1, ii in 0:mr:ib-1
                        aa=pa+ii*kb*sizeof(T)
                        bb=pb+jj*kb*sizeof(T)
                        mi,nj=min(mr,ib-ii),min(nr,jb-jj)
                        i0,j0=ic+ii,jc+jj
                        crossing=false
                        if tri!=0
                            # Upper keeps i<=j, lower i>=j.
                            (tri==1 ? i0>j0+nj-1 : i0+mi-1<j0) && continue
                            crossing=tri==1 ? i0+mi-1>j0 : i0<j0+nj-1
                        end
                        if crossing
                            ep=pointer(edge)
                            T<:Complex ? _call_simd_complex_micro!(ep,mr,aa,bb,kb,α,zero(T),W,R,N) :
                                         _call_simd_micro!(ep,mr,aa,bb,kb,α,zero(T),W,R,N)
                            @inbounds for j in 1:nj, i in 1:mi
                                ci,cj=i0+i-1,j0+j-1
                                (tri==1 ? ci<=cj : ci>=cj) || continue
                                v=edge[i,j]
                                C[ci,cj]=iszero(bp) ? v : v+bp*C[ci,cj]
                            end
                        elseif mi==mr && nj==nr && algorithm===:simd && T <: RealFloat && W>1 && _can_pointer(C)
                            cp=pointer(C)+((ic+ii-1)+(jc+jj-1)*stride(C,2))*sizeof(T)
                            _call_simd_micro!(cp,stride(C,2),aa,bb,kb,α,bp,W,R,N)
                        elseif mi==mr && nj==nr && algorithm===:simd && T <: Complex && _can_pointer(C)
                            cp=pointer(C)+((ic+ii-1)+(jc+jj-1)*stride(C,2))*sizeof(T)
                            _call_simd_complex_micro!(cp,stride(C,2),aa,bb,kb,α,bp,W,R,N)
                        elseif mi==mr && nj==nr && algorithm===:register
                            _call_register_micro!(C,ic+ii,jc+jj,aa,bb,kb,α,bp,mr,nr)
                        elseif edge!==nothing
                            ep=pointer(edge)
                            T<:Complex ? _call_simd_complex_micro!(ep,mr,aa,bb,kb,α,zero(T),W,R,N) :
                                         _call_simd_micro!(ep,mr,aa,bb,kb,α,zero(T),W,R,N)
                            @inbounds for j in 1:nj, i in 1:mi
                                v=edge[i,j]; ci,cj=ic+ii+i-1,jc+jj+j-1
                                # bp=0 must not read C.
                                C[ci,cj]=iszero(bp) ? v : v+bp*C[ci,cj]
                            end
                        else
                            _scalar_micro!(C,ic+ii,jc+jj,aa,bb,kb,α,bp,mr,nr,mi,nj)
                        end
                    end
                end
            end
        end
    end
    C
end
_packed_region!(C::AbstractMatrix{T},A,B,α,β,buf,config,rows,cols,
                W::Int,R::Int,N::Int,algorithm,tri::Integer) where T =
    _packed_region!(C,A,B,α,β,buf,config,rows,cols,W,R,N,algorithm,nothing,tri)

function _workspace_fits(w::GemmWorkspace,m,n,k,config,mr,nr)
    !isempty(w.buffers) && w.acap>=cld(min(m,config.mc),mr)*mr*min(k,config.kc) && w.bcap>=cld(min(n,config.nc),nr)*nr*min(k,config.kc)
end
function _workspace_check(w::GemmWorkspace{T},m,n,k,config,mr,nr) where T
    acap=Base.checked_mul(cld(min(m,config.mc),mr)*mr,min(k,config.kc))
    bcap=Base.checked_mul(cld(min(n,config.nc),nr)*nr,min(k,config.kc))
    w.acap>=acap && w.bcap>=bcap || throw(ArgumentError("workspace too small; construct one for these dimensions and configuration"))
    isempty(w.buffers) && throw(ArgumentError("workspace has no worker buffers"))
    for (index,buffer) in enumerate(w.buffers)
        1<=buffer.ai<=length(buffer.a) && acap<=length(buffer.a)-buffer.ai+1 || throw(ArgumentError("invalid A packing buffer capacity/offset"))
        1<=buffer.bi<=length(buffer.b) && bcap<=length(buffer.b)-buffer.bi+1 || throw(ArgumentError("invalid B packing buffer capacity/offset"))
        Base.mightalias(buffer.a,buffer.b) && throw(ArgumentError("packing panels overlap"))
        for previous in 1:index-1
            other=w.buffers[previous]
            (Base.mightalias(buffer.a,other.a) || Base.mightalias(buffer.a,other.b) ||
             Base.mightalias(buffer.b,other.a) || Base.mightalias(buffer.b,other.b)) && throw(ArgumentError("worker packing buffers overlap"))
        end
    end
    nothing
end
@inline function _check_operand_alias(w::GemmWorkspace, A, check_tiles::Bool)
    for buffer in w.buffers
        (Base.mightalias(buffer.a, A) || Base.mightalias(buffer.b, A)) && throw(ArgumentError("operand aliases owned packing storage"))
    end
    if check_tiles
        for tile in w.tiles
            Base.mightalias(tile, A) && throw(ArgumentError("operand aliases owned triangular tile storage"))
        end
    end
    for tile in w.edge_tiles
        Base.mightalias(tile, A) && throw(ArgumentError("operand aliases owned edge tile storage"))
    end
    nothing
end

function _workspace_operands_check(w::GemmWorkspace,operands...;tiles=false)
    foreach(operands) do A
        _check_operand_alias(w, A, tiles)
    end
    nothing
end

# Sequential time per m*n*k unit, relative to Float32 (complex products are
# four real FMAs; Float64 lanes are half as wide).
_gemm_cost(::Type{T}) where T=(T<:Complex ? 4 : 1)*(sizeof(real(T))÷4)
# One worker per _GEMM_GRAIN cost units (about 100 us of sequential time on
# Apple M5 Max). Every product above config.parallel_threshold used all
# threads: at 256x256 on 12-18 threads it ran slower than on 4, and slower
# than one thread for SYRK/HERK/SYMM. Above the threshold at least two
# workers run.
const _GEMM_GRAIN=6e6
# Workers for `work` m*n*k units; also sizes the panels allocated per call
# when no workspace is supplied (one panel pair per thread cost 68 MB per
# 512x512 call on 18 threads: GC and page faults made the median 5x the
# minimum).
function _gemm_workers(::Type{T},work,threaded,threshold,limit) where T
    threaded && limit>=2 && work>=threshold || return 1
    clamp(floor(Int,work*_gemm_cost(T)/_GEMM_GRAIN),2,limit)
end
# Grid pr x pc <= nt using the most workers, balancing worker count with aspect
# ratio fidelity (avoiding degenerate 1D strips on prime thread counts like 17).
function _worker_grid(nt,m,n,rt,ct)
    target=sqrt(nt*m/n); best=(1,min(nt,ct)); score=-Inf
    for pr in 1:min(nt,rt)
        pc=min(nt÷pr,ct)
        w=pr*pc; w==0 && continue
        s=Float64(w) - 1.5 * abs(log(pr/target))
        s>score && (best=(pr,pc); score=s)
    end
    best
end
function _run_packed!(C,A,B,α,β,w,config,shape,algorithm,threaded,tri=0)
    m,n,k=size(A,1),size(B,2),size(A,2)
    width,r,nr=shape
    mr=width*r
    lock(w.guard) do
        _workspace_check(w,m,n,k,config,mr,nr)
        _workspace_operands_check(w,C,A,B)
        work=Float64(m)*n*k/(tri==0 ? 1 : 2)
        nt=_gemm_workers(eltype(C),work,threaded,config.parallel_threshold,min(length(w.buffers),Threads.nthreads()))
        # General products: a pr x pc grid of output blocks, one per worker,
        # so each worker packs m/pr rows of A and n/pc columns of B (column
        # bands alone made every worker pack all of A). pr follows the shape.
        # Triangles keep column bands: column tile t of a triangle holds work
        # proportional to t (upper) or tiles-t (lower), so equal-work bounds
        # follow a square root.
        rt,ct=cld(m,mr),cld(n,nr)
        # One assignment each: pr and pc are captured by cbound/region.
        pr,pc=tri==0 ? _worker_grid(min(nt,rt*ct),m,n,rt,ct) : (1,min(nt,ct))
        nt=pr*pc
        cbound(c)=tri==1 ? round(Int,ct*sqrt(c/pc)) : tri==2 ? ct-round(Int,ct*sqrt(1-c/pc)) : fld(c*ct,pc)
        region(worker)=begin
            i,j=divrem(worker-1,pc)
            rows=fld(i*rt,pr)*mr+1:min(m,fld((i+1)*rt,pr)*mr)
            cols=cbound(j)*nr+1:min(n,cbound(j+1)*nr)
            (rows,cols)
        end
        needs_edge=algorithm===:simd && _can_pointer(C) && (eltype(C)<:Complex || (eltype(C)<:RealFloat && width>1))
        if needs_edge
            edge_tiles=_workspace_edge_tiles!(w,nt,mr,nr)
            if nt<=1
                _packed_region!(C,A,B,α,β,w.buffers[1],config,1:m,1:n,width,r,nr,algorithm,edge_tiles[1],tri)
            else
                Threads.@threads for worker in 1:nt
                    rows,cols=region(worker)
                    (isempty(rows) || isempty(cols)) && continue
                    _packed_region!(C,A,B,α,β,w.buffers[worker],config,rows,cols,width,r,nr,algorithm,edge_tiles[worker],tri)
                end
            end
        else
            if nt<=1
                _packed_region!(C,A,B,α,β,w.buffers[1],config,1:m,1:n,width,r,nr,algorithm,nothing,tri)
            else
                Threads.@threads for worker in 1:nt
                    rows,cols=region(worker)
                    (isempty(rows) || isempty(cols)) && continue
                    _packed_region!(C,A,B,α,β,w.buffers[worker],config,rows,cols,width,r,nr,algorithm,nothing,tri)
                end
            end
        end
    end
    C
end

# Non-packing algorithms still honor an explicitly shared context's lock and
# ownership contract. Packed algorithms retain their existing task scheduler.
@inline function _unpacked_workspace(f::F,w,::Type{T},C,A,B) where {F,T}
    w===nothing && return f()
    w isa GemmWorkspace{T} || throw(ArgumentError("workspace element type mismatch"))
    lock(w.guard) do
        _workspace_operands_check(w,C,A,B)
        f()
    end
end

"""
    gemm!(C, A, B, α=1, β=0; transA=:N, transB=:N, algorithm=:auto,
          kernel=:auto, threaded=true, workspace=nothing, config=GemmConfig(eltype(C)))

Compute C ← α op(A) op(B) + β C in native Julia. Equal Float32/64 or
ComplexF32/64 element types; 1-based matrices and views. Overlapping inputs
are snapshotted before writing C. β=0 never reads old C; α=0 or K=0 never
reads A/B. Algorithms: reference, blocked, packed, register, simd, auto.
Explicit unavailable ISA requests throw before changing C.
"""
function gemm!(C::AbstractMatrix{T},A::AbstractMatrix{T},B::AbstractMatrix{T},
               α0=1,β0=0; transA=:N,transB=:N,algorithm=:auto,kernel=:auto,
               threaded=true,workspace=nothing,config=GemmConfig(T)) where {T<:NativeFloat}
    isconcretetype(T) || throw(ArgumentError("a concrete native scalar type is required"))
    # Concrete operand types for the usual untransposed call: _op returns a
    # Union (A, transpose, adjoint) and union-typed operands captured by the
    # closures below were boxed at every call.
    if transA === :N && transB === :N
        return _gemm_ops!(C,A,B,α0,β0,algorithm,kernel,threaded,workspace,config)
    end
    kA = _trans_kind(transA)
    kB = _trans_kind(transB)
    if kA == 0
        if kB == 1
            return _gemm_ops!(C, A, transpose(B), α0, β0, algorithm, kernel, threaded, workspace, config)
        elseif kB == 2
            return _gemm_ops!(C, A, adjoint(B), α0, β0, algorithm, kernel, threaded, workspace, config)
        else
            return _gemm_ops!(C, A, B, α0, β0, algorithm, kernel, threaded, workspace, config)
        end
    elseif kA == 1
        if kB == 0
            return _gemm_ops!(C, transpose(A), B, α0, β0, algorithm, kernel, threaded, workspace, config)
        elseif kB == 1
            return _gemm_ops!(C, transpose(A), transpose(B), α0, β0, algorithm, kernel, threaded, workspace, config)
        else
            return _gemm_ops!(C, transpose(A), adjoint(B), α0, β0, algorithm, kernel, threaded, workspace, config)
        end
    else
        if kB == 0
            return _gemm_ops!(C, adjoint(A), B, α0, β0, algorithm, kernel, threaded, workspace, config)
        elseif kB == 1
            return _gemm_ops!(C, adjoint(A), transpose(B), α0, β0, algorithm, kernel, threaded, workspace, config)
        else
            return _gemm_ops!(C, adjoint(A), adjoint(B), α0, β0, algorithm, kernel, threaded, workspace, config)
        end
    end
end
function _gemm_ops!(C::AbstractMatrix{T},Aop,Bop,α0,β0,algorithm,kernel,threaded,workspace,config) where T
    m,n,k=_gemm_dims(C,Aop,Bop)
    α,β=convert(T,α0),convert(T,β0)
    algorithm in (:auto,:reference,:blocked,:packed,:register,:simd) || throw(ArgumentError("unknown algorithm $algorithm"))
    kind=_kernel_kind(kernel)
    (m==0 || n==0) && return C
    if iszero(α) || k==0
        return _unpacked_workspace(()->_scale!(C,β),workspace,T,C,Aop,Bop)
    end
    # Snapshots of operands overlapping C, through a barrier (no
    # union-typed operand in the closures of _gemm_run!).
    ca,cb=_mightalias(C,Aop),_mightalias(C,Bop)
    ca ? (cb ? _gemm_run!(C,copy(Aop),copy(Bop),α,β,m,n,k,algorithm,kind,threaded,workspace,config) :
               _gemm_run!(C,copy(Aop),Bop,α,β,m,n,k,algorithm,kind,threaded,workspace,config)) :
         (cb ? _gemm_run!(C,Aop,copy(Bop),α,β,m,n,k,algorithm,kind,threaded,workspace,config) :
               _gemm_run!(C,Aop,Bop,α,β,m,n,k,algorithm,kind,threaded,workspace,config))
end
function _gemm_run!(C::AbstractMatrix{T},AA,BB,α,β,m,n,k,algorithm,kind,threaded,workspace,config) where T
    small=config.small_threshold>0 && max(m,n)<=config.small_threshold && k<=512
    method=algorithm===:auto ? (small ? :small : :simd) : algorithm
    method===:small && return _unpacked_workspace(()->_small_gemm!(C,AA,BB,α,β),workspace,T,C,AA,BB)
    method===:reference && return _unpacked_workspace(()->_reference!(C,AA,BB,α,β),workspace,T,C,AA,BB)
    method===:blocked && return _unpacked_workspace(()->_blocked!(C,AA,BB,α,β,config),workspace,T,C,AA,BB)
    shape=_shape(T,kind)
    # Scalar fallback preserves a register tile instead of silently using BLAS.
    packed=method===:simd && (kind===:scalar || !_can_pointer(C)) ? :register : method
    workspace===nothing || workspace isa GemmWorkspace{T} || throw(ArgumentError("workspace element type mismatch"))
    _pooled_workspace(T,workspace,m,n,k,config,_gemm_workers(T,Float64(m)*n*k,threaded,config.parallel_threshold,Threads.nthreads())) do w
        _run_packed!(C,AA,BB,α,β,w,config,shape,packed,threaded)
    end
end

gemm_reference!(C,A,B,α=1,β=0;kwargs...) = gemm!(C,A,B,α,β;algorithm=:reference,kwargs...)
gemm_blocked!(C,A,B,α=1,β=0;kwargs...) = gemm!(C,A,B,α,β;algorithm=:blocked,kwargs...)
