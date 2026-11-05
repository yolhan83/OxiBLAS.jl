# SPDX-License-Identifier: Apache-2.0

# Read the selected triangle while packing, without a dense symmetric expansion.
struct _SymmetricOperand{T,A<:AbstractMatrix{T},H} <: AbstractMatrix{T}
    data::A
    upper::Bool
end
_SymmetricOperand(A::AbstractMatrix{T},upper,hermitian) where T =
    _SymmetricOperand{T,typeof(A),hermitian}(A,upper)
Base.size(S::_SymmetricOperand)=size(S.data)
@inline Base.getindex(S::_SymmetricOperand{T,A,H},i::Int,j::Int) where {T,A,H} =
    convert(T,_symget(S.data,i,j,S.upper,H))

function _symm!(C::AbstractMatrix{T},A::AbstractMatrix{T},B::AbstractMatrix{T},α,β,side,uplo,hermitian;kwargs...) where {T<:GenericFloat}
    n=_square(A); upper=_uplo(uplo)
    Base.require_one_based_indexing(C,B)
    side in (:L,:R) || throw(ArgumentError("side must be L or R"))
    n==(side===:L ? size(B,1) : size(B,2)) || throw(DimensionMismatch("SYMM dimensions differ"))
    size(C)==size(B) || throw(DimensionMismatch("SYMM output dimensions differ"))
    S=_SymmetricOperand(A,upper,hermitian)
    # Snapshot logical entries only: the unused triangle is never inspected.
    aa=!iszero(α) && Base.mightalias(C,A) ? copy(S) : S
    side===:L ? gemm!(C,aa,B,α,β;kwargs...) : gemm!(C,B,aa,α,β;kwargs...)
end
symm!(C,A,B,α=one(eltype(C)),β=zero(eltype(C));side=:L,uplo=:U,kwargs...)=_symm!(C,A,B,α,β,side,uplo,false;kwargs...)
hemm!(C,A,B,α=one(eltype(C)),β=zero(eltype(C));side=:L,uplo=:U,kwargs...)=_symm!(C,A,B,α,β,side,uplo,true;kwargs...)

function _l3_validate(C,algorithm,kernel,blocksize)
    Base.require_one_based_indexing(C)
    _check_output_layout(C)
    blocksize>0 || throw(ArgumentError("blocksize must be positive"))
    algorithm in (:auto,:reference,:blocked,:packed,:register,:simd) || throw(ArgumentError("unknown algorithm $algorithm"))
    _kernel_kind(kernel)
end

# A single pair of packing panels is reused throughout a structured operation.
function _l3_options(::Type{T},m,n,k,algorithm,kernel,threaded,workspace,config;workers=threaded ? Threads.nthreads() : 1) where T
    w=workspace
    if !(algorithm in (:reference,:blocked))
        w===nothing && (w=GemmWorkspace(T,m,n,k;config,workers))
        w isa GemmWorkspace{T} || throw(ArgumentError("workspace element type mismatch"))
        width,r,nr=_shape(T,_kernel_kind(kernel))
        _workspace_check(w,m,n,k,config,width*r,nr)
    end
    (;algorithm,kernel,threaded,workspace=w,config)
end

function _tri_snapshot(A,upper,unit)
    n=size(A,1); S=zeros(eltype(A),n,n)
    @inbounds for j in 1:n, i in (upper ? (1:j) : (j:n))
        S[i,j]=unit && i==j ? one(eltype(A)) : A[i,j]
    end
    S
end

# Triangular products hold n^2*rhs/2 multiply-adds, but their recursive
# small GEMMs run at about half the GEMM rate at n<=256, so the GEMM worker
# rule (_gemm_workers) is applied to n^2*rhs (with n^2*rhs/2, 256x256 Float64
# got 2 workers and ran 1.2x slower than the previous all-thread schedule).
_tri_workers(::Type{T},n,rhs,threaded,config) where T=_gemm_workers(T,Float64(n)^2*rhs,threaded,config.parallel_threshold,min(rhs,Threads.nthreads()))
# Pooled panels (_pooled_workspace) for the recursive products; the
# reference and blocked algorithms use none.
function _tri_pooled(f,::Type{T},B,n,left,algorithm,threaded,workspace,config) where T
    (workspace!==nothing || algorithm in (:reference,:blocked)) && return f(workspace)
    h=cld(n,2); m,p=left ? (h,size(B,2)) : (size(B,1),h)
    _pooled_workspace(f,T,nothing,m,p,h,config,_tri_workers(T,n,left ? size(B,2) : size(B,1),threaded,config))
end
function _tri_options(::Type{T},B,n,left,algorithm,kernel,threaded,workspace,config) where T
    h=cld(n,2)
    m,p=left ? (h,size(B,2)) : (size(B,1),h)
    _l3_options(T,m,p,h,algorithm,kernel,threaded,workspace,config;workers=_tri_workers(T,n,left ? size(B,2) : size(B,1),threaded,config))
end

# In-place leaf traversal preserves every input entry until its last use.
function _trmm_leaf_generic!(B,A,α,left,upper,unit)
    m,n=size(B)
    if left
        @inbounds for j in 1:n, i in (upper ? (1:m) : (m:-1:1))
            s=unit ? B[i,j] : A[i,i]*B[i,j]
            for p in (upper ? (i+1:m) : (1:i-1))
                s=_native_madd(A[i,p],B[p,j],s)
            end
            B[i,j]=α*s
        end
    else
        @inbounds for j in (upper ? (n:-1:1) : (1:n)), i in 1:m
            s=unit ? B[i,j] : B[i,j]*A[j,j]
            for p in (upper ? (1:j-1) : (j+1:n))
                s=_native_madd(B[i,p],A[p,j],s)
            end
            B[i,j]=α*s
        end
    end
    B
end

function _trmm_blocks!(B,A,α,left,upper,unit,bs,options)
    n=size(A,1)
    n<=bs && return _trmm_leaf!(B,A,α,left,upper,unit;kernel=options.kernel,optimized=options.algorithm!==:reference)
    h=n÷2; lo=1:h; hi=h+1:n
    A1=view(A,lo,lo); A2=view(A,hi,hi)
    B1=left ? view(B,lo,:) : view(B,:,lo)
    B2=left ? view(B,hi,:) : view(B,:,hi)
    if left && upper
        _trmm_blocks!(B1,A1,α,left,upper,unit,bs,options)
        gemm!(B1,view(A,lo,hi),B2,α,one(α);options...)
        _trmm_blocks!(B2,A2,α,left,upper,unit,bs,options)
    elseif left
        _trmm_blocks!(B2,A2,α,left,upper,unit,bs,options)
        gemm!(B2,view(A,hi,lo),B1,α,one(α);options...)
        _trmm_blocks!(B1,A1,α,left,upper,unit,bs,options)
    elseif upper
        _trmm_blocks!(B2,A2,α,left,upper,unit,bs,options)
        gemm!(B2,B1,view(A,lo,hi),α,one(α);options...)
        _trmm_blocks!(B1,A1,α,left,upper,unit,bs,options)
    else
        _trmm_blocks!(B1,A1,α,left,upper,unit,bs,options)
        gemm!(B1,B2,view(A,hi,lo),α,one(α);options...)
        _trmm_blocks!(B2,A2,α,left,upper,unit,bs,options)
    end
    B
end

"""In-place triangular multiplication; recursive blocks use native GEMM.
`blocksize` sets the scalar leaf size. GEMM options control off-diagonal updates.
Only aliased A requires a full logical snapshot; B is never fully copied.
"""
function trmm!(B::AbstractMatrix{T},A::AbstractMatrix{T},α0=one(T);side=:L,uplo=:U,trans=:N,unit=false,
               blocksize=8,algorithm=:auto,kernel=:auto,threaded=true,workspace=nothing,config=GemmConfig(T)) where {T<:NativeFloat}
    AA,n,upper=_tri_setup(A,trans,uplo,unit)
    side in (:L,:R) || throw(ArgumentError("side must be L or R"))
    n==(side===:L ? size(B,1) : size(B,2)) || throw(DimensionMismatch("TRMM dimensions differ"))
    _l3_validate(B,algorithm,kernel,blocksize)
    α=convert(T,α0)
    _tri_workspace_call(workspace,B,A) do
        (iszero(α) || isempty(B)) && return _scale!(B,α)
        aa=Base.mightalias(A,B) ? _tri_snapshot(AA,upper,unit) : AA
        n<=blocksize && return _trmm_leaf!(B,aa,α,side===:L,upper,unit;kernel,optimized=algorithm!==:reference)
        _tri_pooled(T,B,n,side===:L,algorithm,threaded,workspace,config) do w
            options=_tri_options(T,B,n,side===:L,algorithm,kernel,threaded,w,config)
            _tri_execute!(B,aa,α,side===:L,upper,unit,blocksize,options,false)
        end
    end
end

function _trsm_leaf_generic!(B,A,left,upper,unit)
    m,n=size(B)
    if left
        @inbounds for j in 1:n, i in (upper ? (m:-1:1) : (1:m))
            s=B[i,j]
            for p in (upper ? (i+1:m) : (1:i-1))
                s=_native_madd(-A[i,p],B[p,j],s)
            end
            B[i,j]=unit ? s : s/A[i,i]
        end
    else
        # Contiguous column updates rather than one strided solve per row.
        @inbounds for j in (upper ? (1:n) : (n:-1:1))
            if !unit
                for i in 1:m
                    B[i,j]/=A[j,j]
                end
            end
            for p in (upper ? (j+1:n) : (1:j-1))
                a=-A[j,p]
                @simd for i in 1:m
                    B[i,p]=_native_madd(B[i,j],a,B[i,p])
                end
            end
        end
    end
    B
end

function _trsm_blocks!(B,A,left,upper,unit,bs,options)
    n=size(A,1)
    n<=bs && return _trsm_leaf!(B,A,left,upper,unit;kernel=options.kernel,optimized=options.algorithm!==:reference)
    h=n÷2; lo=1:h; hi=h+1:n
    A1=view(A,lo,lo); A2=view(A,hi,hi)
    B1=left ? view(B,lo,:) : view(B,:,lo)
    B2=left ? view(B,hi,:) : view(B,:,hi)
    α=one(eltype(B))
    if left && upper
        _trsm_blocks!(B2,A2,left,upper,unit,bs,options)
        gemm!(B1,view(A,lo,hi),B2,-α,α;options...)
        _trsm_blocks!(B1,A1,left,upper,unit,bs,options)
    elseif left
        _trsm_blocks!(B1,A1,left,upper,unit,bs,options)
        gemm!(B2,view(A,hi,lo),B1,-α,α;options...)
        _trsm_blocks!(B2,A2,left,upper,unit,bs,options)
    elseif upper
        _trsm_blocks!(B1,A1,left,upper,unit,bs,options)
        gemm!(B2,B1,view(A,lo,hi),-α,α;options...)
        _trsm_blocks!(B2,A2,left,upper,unit,bs,options)
    else
        _trsm_blocks!(B2,A2,left,upper,unit,bs,options)
        gemm!(B1,B2,view(A,hi,lo),-α,α;options...)
        _trsm_blocks!(B1,A1,left,upper,unit,bs,options)
    end
    B
end

"""Triangular solve with recursive native-GEMM updates and reusable panels.
A singular nonunit diagonal is reported before any mutation of B.
"""
function trsm!(B::AbstractMatrix{T},A::AbstractMatrix{T},α0=one(T);side=:L,uplo=:U,trans=:N,unit=false,
               blocksize=8,algorithm=:auto,kernel=:auto,threaded=true,workspace=nothing,config=GemmConfig(T)) where {T<:NativeFloat}
    AA,n,upper=_tri_setup(A,trans,uplo,unit)
    side in (:L,:R) || throw(ArgumentError("side must be L or R"))
    n==(side===:L ? size(B,1) : size(B,2)) || throw(DimensionMismatch("TRSM dimensions differ"))
    _l3_validate(B,algorithm,kernel,blocksize)
    α=convert(T,α0)
    _tri_workspace_call(workspace,B,A) do
        (iszero(α) || isempty(B)) && return _scale!(B,α)
        aa=Base.mightalias(A,B) ? _tri_snapshot(AA,upper,unit) : AA
        if !unit
            @inbounds for i in 1:n
                iszero(aa[i,i]) && throw(SingularException(i))
            end
        end
        if n<=blocksize
            _scale!(B,α)
            return _trsm_leaf!(B,aa,side===:L,upper,unit;kernel,optimized=algorithm!==:reference)
        end
        _tri_pooled(T,B,n,side===:L,algorithm,threaded,workspace,config) do w
            options=_tri_options(T,B,n,side===:L,algorithm,kernel,threaded,w,config)
            _tri_execute!(B,aa,α,side===:L,upper,unit,blocksize,options,true)
        end
    end
end

function _tri_execute!(B,A,α,left,upper,unit,bs,options,solving)
    rhs=left ? size(B,2) : size(B,1)
    w=options.workspace
    # One assignment: workers is captured by run().
    wanted=_tri_workers(eltype(B),size(A,1),rhs,options.threaded,options.config)
    workers=w isa GemmWorkspace ? min(wanted,length(w.buffers)) : wanted
    if workers<=1
        solving && _scale!(B,α)
        return solving ? _trsm_blocks!(B,A,left,upper,unit,bs,options) : _trmm_blocks!(B,A,α,left,upper,unit,bs,options)
    end
    function run()
        children=w isa GemmWorkspace ? _workspace_children!(w,workers) : fill(nothing,workers)
        _foreach_range(rhs,workers) do range,worker
            out=left ? view(B,:,range) : view(B,range,:)
            opts=merge(options,(;threaded=false,workspace=children[worker]))
            if solving
                _scale!(out,α)
                _trsm_blocks!(out,A,left,upper,unit,bs,opts)
            else
                _trmm_blocks!(out,A,α,left,upper,unit,bs,opts)
            end
        end
        B
    end
    w isa GemmWorkspace ? lock(run,w.guard) : run()
end

function _triangle_scale!(C,β,upper,hermitian)
    n=size(C,1)
    @inbounds for j in 1:n, i in (upper ? (1:j) : (j:n))
        v=iszero(β) ? zero(eltype(C)) : β*(hermitian && i==j ? real(C[i,j]) : C[i,j])
        C[i,j]=hermitian && i==j ? real(v) : v
    end
    C
end

function _triangle_leaf!(C,A,B,D,E,α,γ,β,upper,hermitian,rank2)
    n=size(C,1); k=size(A,2)
    @inbounds for j in 1:n, i in (upper ? (1:j) : (j:n))
        s=zero(eltype(C)); t=zero(eltype(C))
        for p in 1:k
            s=muladd(A[i,p],B[p,j],s)
            rank2 && (t=muladd(D[i,p],E[p,j],t))
        end
        v=α*s+(rank2 ? γ*t : zero(s))
        !iszero(β) && (v+=β*(hermitian && i==j ? real(C[i,j]) : C[i,j]))
        C[i,j]=hermitian && i==j ? real(v) : v
    end
    C
end

function _triangle_jobs(n,bs,upper)
    jobs=Tuple{Int,Int}[]
    for jc in 1:bs:n, ic in (upper ? (1:bs:jc) : (jc:bs:n))
        push!(jobs,(ic,jc))
    end
    jobs
end

function _triangle_worker_count(n,k,bs,options,rank2)
    options.threaded || return 1
    work=Float64(n)*n*k*(rank2 ? 2 : 1)
    work>=options.config.parallel_threshold || return 1
    blocks=cld(n,bs); tiles=blocks*(blocks+1)÷2
    capacity=options.workspace isa GemmWorkspace ? length(options.workspace.buffers) : Threads.nthreads()
    min(Threads.nthreads(),capacity,tiles)
end

function _triangle_tile!(C,A,B,D,E,α,γ,β,upper,hermitian,rank2,bs,options,tile,ic,jc)
    n=size(C,1); cols=jc:min(n,jc+bs-1); rows=ic:min(n,ic+bs-1)
    diagonal=ic==jc
    out=diagonal ? view(tile,1:length(rows),1:length(cols)) : view(C,rows,cols)
    gemm!(out,view(A,rows,:),view(B,:,cols),α,diagonal ? zero(β) : β;options...)
    rank2 && gemm!(out,view(D,rows,:),view(E,:,cols),γ,one(β);options...)
    if diagonal
        @inbounds for j in eachindex(cols), i in (upper ? (1:j) : (j:length(rows)))
            ii,jj=rows[i],cols[j]
            v=out[i,j]
            !iszero(β) && (v+=β*(hermitian && i==j ? real(C[ii,jj]) : C[ii,jj]))
            C[ii,jj]=hermitian && i==j ? real(v) : v
        end
    end
    nothing
end

function _triangle_task!(C,A,B,D,E,α,γ,β,upper,hermitian,rank2,bs,options,tile,jobs,worker,workers)
    for idx in worker:workers:length(jobs)
        ic,jc=jobs[idx]
        _triangle_tile!(C,A,B,D,E,α,γ,β,upper,hermitian,rank2,bs,options,tile,ic,jc)
    end
    nothing
end

function _triangle_parallel!(C,A,B,D,E,α,γ,β,upper,hermitian,rank2,bs,options,workers)
    n=size(C,1); w=options.workspace
    # Buffer ownership follows the task, not threadid(). The parent guard
    # excludes other calls sharing w for the entire tile schedule. Children
    # lock their own single-buffer wrapper and never re-enter the parent lock.
    if w isa GemmWorkspace
        _workspace_operands_check(w,C,A,B,D,E;tiles=true)
        if w.jobkey!=(n,bs,upper)
            empty!(w.jobs); append!(w.jobs,_triangle_jobs(n,bs,upper)); w.jobkey=(n,bs,upper)
        end
        jobs=w.jobs
        children=_workspace_children!(w,workers)
        tiles=_workspace_tiles!(w,workers,min(bs,n))
    else
        jobs=_triangle_jobs(n,bs,upper)
        children=fill(nothing,workers)
        tiles=[Matrix{eltype(C)}(undef,min(bs,n),min(bs,n)) for _ in 1:workers]
    end
    # jobs is assigned in both branches: rebound before the spawned closures.
    @sync for worker in 1:workers
        let worker=worker, opts=merge(options,(;workspace=children[worker],threaded=false)), tile=tiles[worker], jobs=jobs
            Threads.@spawn _triangle_task!(C,A,B,D,E,α,γ,β,upper,hermitian,rank2,bs,opts,tile,jobs,worker,workers)
        end
    end
    C
end

function _triangle_products!(C,A,B,D,E,α,γ,β,upper,hermitian,rank2,bs,options)
    n=size(C,1)
    workers=_triangle_worker_count(n,size(A,2),bs,options,rank2)
    if workers>1
        if options.workspace isa GemmWorkspace
            return lock(options.workspace.guard) do
                _triangle_parallel!(C,A,B,D,E,α,γ,β,upper,hermitian,rank2,bs,options,workers)
            end
        end
        return _triangle_parallel!(C,A,B,D,E,α,γ,β,upper,hermitian,rank2,bs,options,workers)
    end
    if options.workspace isa GemmWorkspace
        return lock(options.workspace.guard) do
            _workspace_operands_check(options.workspace,C,A,B,D,E;tiles=true)
            # Local to the lock block: the outer `tile` below is a separate binding.
            local wtile=_workspace_tiles!(options.workspace,1,min(bs,n))[1]
            # The calling task may re-enter its own ReentrantLock in GEMM.
            for jc in 1:bs:n, ic in (upper ? (1:bs:jc) : (jc:bs:n))
                _triangle_tile!(C,A,B,D,E,α,γ,β,upper,hermitian,rank2,bs,options,wtile,ic,jc)
            end
            C
        end
    end
    tile=Matrix{eltype(C)}(undef,min(bs,n),min(bs,n))
    # No outer task schedule here. A single large tile may still use the
    # existing inner GEMM parallelism; multi-task schedules always disable it.
    opts=options
    for jc in 1:bs:n, ic in (upper ? (1:bs:jc) : (jc:bs:n))
        _triangle_tile!(C,A,B,D,E,α,γ,β,upper,hermitian,rank2,bs,opts,tile,ic,jc)
    end
    C
end

# Triangle-restricted packed GEMM (one or two products into the selected
# triangle of C): micro tiles below/above the diagonal are skipped inside the
# packed GEMM loop, so A and B are packed as in one GEMM. Tiles of `blocksize`
# each calling GEMM repacked both panels per tile and ran 1.3-3.7x behind
# OpenBLAS (M5 Max, n=64-2048). Returns nothing when the request does not
# select the SIMD packed kernel or the supplied workspace is too small; the
# tile schedule below then applies.
function _triangle_packed!(C::AbstractMatrix{T},A,B,D,E,α,γ,β,upper,hermitian,rank2,algorithm,kernel,threaded,workspace,config) where T
    algorithm in (:auto,:simd) && _can_pointer(C) || return nothing
    kind=_kernel_kind(kernel); kind===:scalar && return nothing
    width,r,nr=_shape(T,kind)
    T<:RealFloat && width<=1 && return nothing
    n=size(C,1); k=size(A,2)
    workspace===nothing || workspace isa GemmWorkspace{T} || throw(ArgumentError("workspace element type mismatch"))
    workspace===nothing || _workspace_fits(workspace,n,n,k,config,width*r,nr) || return nothing
    # β=1 leaves the triangle unchanged (the Hermitian diagonal is made real below).
    isone(β) || _triangle_scale!(C,β,upper,hermitian)
    tri=upper ? 1 : 2
    _pooled_workspace(T,workspace,n,n,k,config,_gemm_workers(T,Float64(n)*n*k/2,threaded,config.parallel_threshold,Threads.nthreads())) do w
        _run_packed!(C,A,B,α,one(T),w,config,(width,r,nr),:simd,threaded,tri)
        rank2 && _run_packed!(C,D,E,γ,one(T),w,config,(width,r,nr),:simd,threaded,tri)
    end
    if hermitian
        @inbounds for j in 1:n
            C[j,j]=real(C[j,j])
        end
    end
    C
end

function _rankk!(C::AbstractMatrix{T},A::AbstractMatrix{T},B::AbstractMatrix{T},α,β,uplo,trans,hermitian,rank2;
                 blocksize=128,algorithm=:auto,kernel=:auto,threaded=true,workspace=nothing,config=GemmConfig(T)) where {T<:NativeFloat}
    n=_square(C); upper=_uplo(uplo)
    hermitian && !(trans in (:N,'N',:C,'C')) && throw(ArgumentError("Hermitian rank-k accepts N or C"))
    !hermitian && !(trans in (:N,'N',:T,'T')) && throw(ArgumentError("symmetric rank-k accepts N or T"))
    hermitian && (!isreal(β) || (!rank2 && !isreal(α))) && throw(ArgumentError("HERK alpha/beta and HER2K beta must be real"))
    aa=_op(A,trans); bb=_op(B,trans)
    Base.require_one_based_indexing(aa,bb)
    size(aa,1)==n && size(bb)==size(aa) || throw(DimensionMismatch("rank-k dimensions differ"))
    _l3_validate(C,algorithm,kernel,blocksize)
    α,β=convert(T,α),convert(T,β)
    (iszero(α) || size(aa,2)==0 || n==0) && return _triangle_scale!(C,β,upper,hermitian)
    aa=Base.mightalias(C,aa) ? copy(aa) : aa
    bb=A===B ? aa : (Base.mightalias(C,bb) ? copy(bb) : bb)
    rhs=hermitian ? adjoint(bb) : transpose(bb)
    rhs2=hermitian ? adjoint(aa) : transpose(aa)
    γ=hermitian ? conj(α) : α
    n<=16 && return _triangle_leaf!(C,aa,rhs,bb,rhs2,α,γ,β,upper,hermitian,rank2)
    _triangle_packed!(C,aa,rhs,bb,rhs2,α,γ,β,upper,hermitian,rank2,algorithm,kernel,threaded,workspace,config)===nothing || return C
    options=_l3_options(T,min(blocksize,n),min(blocksize,n),size(aa,2),algorithm,kernel,threaded,workspace,config)
    _triangle_products!(C,aa,rhs,bb,rhs2,α,γ,β,upper,hermitian,rank2,blocksize,options)
end
syrk!(C,A,α=one(eltype(C)),β=zero(eltype(C));uplo=:U,trans=:N,kwargs...)=_rankk!(C,A,A,α,β,uplo,trans,false,false;kwargs...)
syr2k!(C,A,B,α=one(eltype(C)),β=zero(eltype(C));uplo=:U,trans=:N,kwargs...)=_rankk!(C,A,B,α,β,uplo,trans,false,true;kwargs...)
herk!(C,A,α=one(real(eltype(C))),β=zero(real(eltype(C)));uplo=:U,trans=:N,kwargs...)=_rankk!(C,A,A,α,β,uplo,trans,true,false;kwargs...)
her2k!(C,A,B,α=one(eltype(C)),β=zero(real(eltype(C)));uplo=:U,trans=:N,kwargs...)=_rankk!(C,A,B,α,β,uplo,trans,true,true;kwargs...)

"""
    gemmt!(C,A,B,α=1,β=0;uplo=:U,transA=:N,transB=:N,blocksize=128,...)

C[triangle] ← α op(A) op(B) + β C[triangle], preserving the other triangle.
Rectangular inputs, N/T/C, alias snapshots and GEMM options are supported.
Only diagonal tiles use temporary output storage, bounded by blocksize².
"""
function gemmt!(C::AbstractMatrix{T},A::AbstractMatrix{T},B::AbstractMatrix{T},α=one(T),β=zero(T);
                uplo=:U,transA=:N,transB=:N,blocksize=128,algorithm=:auto,kernel=:auto,
                threaded=true,workspace=nothing,config=GemmConfig(T)) where {T<:NativeFloat}
    n=_square(C); upper=_uplo(uplo)
    aa,bb=_op(A,transA),_op(B,transB)
    _gemm_dims(C,aa,bb)
    _l3_validate(C,algorithm,kernel,blocksize)
    α,β=convert(T,α),convert(T,β)
    (iszero(α) || size(aa,2)==0 || n==0) && return _triangle_scale!(C,β,upper,false)
    aa=Base.mightalias(C,aa) ? copy(aa) : aa
    bb=Base.mightalias(C,bb) ? copy(bb) : bb
    n<=16 && return _triangle_leaf!(C,aa,bb,aa,bb,α,zero(T),β,upper,false,false)
    _triangle_packed!(C,aa,bb,aa,bb,α,zero(T),β,upper,false,false,algorithm,kernel,threaded,workspace,config)===nothing || return C
    options=_l3_options(T,min(blocksize,n),min(blocksize,n),size(aa,2),algorithm,kernel,threaded,workspace,config)
    _triangle_products!(C,aa,bb,aa,bb,α,zero(T),β,upper,false,false,blocksize,options)
end
