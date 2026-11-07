# SPDX-License-Identifier: Apache-2.0
_generic_accum(::Type{T}) where T=T===Float16 ? Float32 : T===Complex{Float16} ? ComplexF32 : T
"""
    GenericWorkspace(::Type{T}; blocksize=32, workers=...)

Reusable tile accumulation buffers for extended GEMM. A shared context is
serialized; different worker buffers belong to different tasks. Float16 uses
Float32 accumulators. Arbitrary floating types never enter native SIMD pointer kernels.
"""
mutable struct GenericWorkspace{T,W}
    buffers::Vector{Matrix{W}}
    blocksize::Int
    guard::ReentrantLock
    children::Vector{GenericWorkspace{T,W}}
end
function GenericWorkspace(::Type{T};blocksize::Integer=32,workers::Integer=generic_threadsafe(T) ? Threads.nthreads() : 1) where {T<:GenericFloat}
    blocksize>=1 && workers>=1 || throw(ArgumentError("positive tile size and worker count required"))
    _generic_require(T)
    b=Int(blocksize); W=_generic_accum(T)
    GenericWorkspace{T,W}([Matrix{W}(undef,b,b) for _ in 1:workers],b,ReentrantLock(),GenericWorkspace{T,W}[])
end
function GemmWorkspace(::Type{T},m::Integer,n::Integer,k::Integer;config=GemmConfig(T),workers=generic_threadsafe(T) ? Threads.nthreads() : 1) where {T<:GenericFloat}
    min(m,n,k)>=0 || throw(ArgumentError("negative workspace dimension"))
    GenericWorkspace(T;blocksize=max(1,min(32,config.mc,config.nc,max(m,n))),workers)
end
function _workspace_children!(workspace::GenericWorkspace{T,W},workers) where {T,W}
    workers<=length(workspace.buffers) || throw(DimensionMismatch("insufficient extended worker buffers"))
    while length(workspace.children)<workers
        i=length(workspace.children)+1
        push!(workspace.children,GenericWorkspace{T,W}([workspace.buffers[i]],workspace.blocksize,ReentrantLock(),GenericWorkspace{T,W}[]))
    end
    workspace.children
end
function _generic_options(algorithm,kernel,blocksize)
    algorithm in (:auto,:reference,:blocked) || throw(ArgumentError("extended BLAS supports auto/reference/blocked scalar algorithms"))
    kernel in (:auto,:scalar) || throw(ArgumentError("extended BLAS has no native SIMD ISA kernel"))
    blocksize>=1 || throw(ArgumentError("positive extended tile size required"))
end
function gemm!(C::AbstractMatrix{T},A::AbstractMatrix{T},B::AbstractMatrix{T},alpha0=1,beta0=0;
               transA=:N,transB=:N,algorithm=:auto,kernel=:auto,threaded=true,workspace=nothing,config=GemmConfig(T)) where {T<:GenericFloat}
    _generic_require(T)
    threaded=threaded && generic_threadsafe(T)
    AA,BB=_op(A,transA),_op(B,transB); m,n,k=_gemm_dims(C,AA,BB)
    _generic_options(algorithm,kernel,1); alpha,beta=convert(T,alpha0),convert(T,beta0)
    (m==0 || n==0) && return C
    if iszero(alpha) || k==0
        workspace===nothing && return _scale!(C,beta)
        workspace isa GenericWorkspace{T} || throw(ArgumentError("generic workspace type differs"))
        return lock(workspace.guard) do
            any(tile->Base.mightalias(tile,C) || Base.mightalias(tile,AA) || Base.mightalias(tile,BB),workspace.buffers) &&
                throw(ArgumentError("operand aliases owned accumulation storage"))
            _scale!(C,beta)
        end
    end
    wanted=_operation_workers(cld(m,32)*cld(n,32),Float64(m)*n*k,threaded,config.parallel_threshold)
    w=workspace===nothing ? GenericWorkspace(T;blocksize=min(32,max(m,n)),workers=wanted) : workspace
    w isa GenericWorkspace{T} || throw(ArgumentError("extended workspace type differs"))
    lock(w.guard) do
        any(tile->Base.mightalias(tile,C) || Base.mightalias(tile,AA) || Base.mightalias(tile,BB),w.buffers) &&
            throw(ArgumentError("operand aliases owned extended accumulation storage"))
        aa=Base.mightalias(C,AA) ? copy(AA) : AA; bb=Base.mightalias(C,BB) ? copy(BB) : BB
        bs=w.blocksize; rowtiles=cld(m,bs); jobs=rowtiles*cld(n,bs); W=_generic_accum(T)
        aw,bw=W(alpha),W(beta)
        workers=min(length(w.buffers),_operation_workers(jobs,Float64(m)*n*k,threaded,config.parallel_threshold))
        _foreach_range(jobs,workers) do range,worker
            tile=w.buffers[worker]
            for job in range
                firstrow=mod(job-1,rowtiles)*bs+1; firstcol=fld(job-1,rowtiles)*bs+1
                rows=min(bs,m-firstrow+1); cols=min(bs,n-firstcol+1)
                for j in 1:cols, i in 1:rows
                    tile[i,j]=zero(W)
                end
                for p in 1:k, j in 1:cols
                    value=W(bb[p,firstcol+j-1])
                    for i in 1:rows
                        tile[i,j]=_arithmetic_madd(W(aa[firstrow+i-1,p]),value,tile[i,j])
                    end
                end
                for j in 1:cols, i in 1:rows
                    value=aw*tile[i,j]
                    !iszero(beta) && (value+=bw*W(C[firstrow+i-1,firstcol+j-1]))
                    C[firstrow+i-1,firstcol+j-1]=convert(T,value)
                end
            end
        end
        C
    end
end
function trsm!(B::AbstractMatrix{T},A::AbstractMatrix{T},alpha=one(T);side=:L,uplo=:U,trans=:N,unit=false,
               blocksize=32,algorithm=:auto,kernel=:auto,threaded=true,workspace=nothing,config=GemmConfig(T)) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    AA,n,upper=_tri_setup(A,trans,uplo,unit); _check_output_layout(B); _generic_options(algorithm,kernel,blocksize)
    side in (:L,:R) || throw(ArgumentError("side must be L or R"))
    n==(side===:L ? size(B,1) : size(B,2)) || throw(DimensionMismatch("extended TRSM dimensions differ"))
    alpha=convert(T,alpha); (iszero(alpha) || isempty(B)) && return _scale!(B,alpha)
    aa=Base.mightalias(B,A) ? _tri_snapshot(AA,upper,unit) : AA
    !unit && foreach(i->iszero(aa[i,i]) && throw(SingularException(i)),1:n)
    W=_generic_accum(T); X=W===T ? B : W.(B)
    # Copy only the selected logical triangle when widening; never read the
    # unstored entries or a unit diagonal, even if poisoned by NaN.
    operand=W===T ? aa : W.(_tri_snapshot(aa,upper,unit))
    _scale!(X,W(alpha)); _trsm_leaf!(X,operand,side===:L,upper,unit;kernel,optimized=algorithm!==:reference)
    X===B || copyto!(B,X); B
end
function trmm!(B::AbstractMatrix{T},A::AbstractMatrix{T},alpha=one(T);side=:L,uplo=:U,trans=:N,unit=false,
               blocksize=32,algorithm=:auto,kernel=:auto,threaded=true,workspace=nothing,config=GemmConfig(T)) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    AA,n,upper=_tri_setup(A,trans,uplo,unit); _check_output_layout(B); _generic_options(algorithm,kernel,blocksize)
    side in (:L,:R) || throw(ArgumentError("side must be L or R"))
    n==(side===:L ? size(B,1) : size(B,2)) || throw(DimensionMismatch("extended TRMM dimensions differ"))
    alpha=convert(T,alpha); (iszero(alpha) || isempty(B)) && return _scale!(B,alpha)
    aa=Base.mightalias(B,A) ? _tri_snapshot(AA,upper,unit) : AA
    W=_generic_accum(T); X=W===T ? B : W.(B); operand=W===T ? aa : W.(_tri_snapshot(aa,upper,unit))
    _trmm_leaf!(X,operand,W(alpha),side===:L,upper,unit;kernel,optimized=algorithm!==:reference)
    X===B || copyto!(B,X); B
end
function _rankk!(C::AbstractMatrix{T},A::AbstractMatrix{T},B::AbstractMatrix{T},alpha0,beta0,uplo,trans,hermitian,rank2;
                 blocksize=32,algorithm=:auto,kernel=:auto,threaded=true,workspace=nothing,config=GemmConfig(T)) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    n=_square(C); upper=_uplo(uplo); tr=_trans_kind(trans)
    (hermitian ? tr in (0,2) : tr in (0,1)) || throw(ArgumentError("invalid structured transpose"))
    hermitian && (!isreal(beta0) || (!rank2 && !isreal(alpha0))) && throw(ArgumentError("Hermitian coefficients must be real"))
    opa,opb=_op(A,trans),_op(B,trans); size(opa,1)==n && size(opb)==size(opa) || throw(DimensionMismatch("extended rank-k dimensions differ"))
    _generic_options(algorithm,kernel,blocksize); _check_output_layout(C)
    alpha,beta=convert(T,alpha0),convert(T,beta0)
    (iszero(alpha) || size(opa,2)==0 || n==0) && return _triangle_scale!(C,beta,upper,hermitian)
    # aa/bb are assigned once: the parallel closure below captures them.
    aa=Base.mightalias(C,opa) ? copy(opa) : opa; bb=A===B ? aa : Base.mightalias(C,opb) ? copy(opb) : opb
    W=_generic_accum(T); rhs=hermitian ? adjoint(bb) : transpose(bb); rhs2=hermitian ? adjoint(aa) : transpose(aa)
    workers=_operation_workers(n,Float64(n)*n*size(aa,2),threaded,config.parallel_threshold)
    _foreach_range(n,workers) do columns,_
        for j in columns, i in (upper ? (1:j) : (j:n))
            s=zero(W); t=zero(W)
            for p in axes(aa,2)
                s=_arithmetic_madd(W(aa[i,p]),W(rhs[p,j]),s)
                rank2 && (t=_arithmetic_madd(W(bb[i,p]),W(rhs2[p,j]),t))
            end
            value=W(alpha)*s+(rank2 ? W(hermitian ? conj(alpha) : alpha)*t : zero(W))
            !iszero(beta) && (value+=W(beta)*W(hermitian && i==j ? real(C[i,j]) : C[i,j]))
            C[i,j]=convert(T,hermitian && i==j ? real(value) : value)
        end
    end
    C
end
function gemmt!(C::AbstractMatrix{T},A::AbstractMatrix{T},B::AbstractMatrix{T},alpha=one(T),beta=zero(T);
                uplo=:U,transA=:N,transB=:N,blocksize=32,algorithm=:auto,kernel=:auto,threaded=true,workspace=nothing,config=GemmConfig(T)) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    aa,bb=_op(A,transA),_op(B,transB); n=_square(C); _gemm_dims(C,aa,bb); upper=_uplo(uplo)
    _generic_options(algorithm,kernel,blocksize); alpha,beta=convert(T,alpha),convert(T,beta)
    (iszero(alpha) || size(aa,2)==0) && return _triangle_scale!(C,beta,upper,false)
    aa=Base.mightalias(C,aa) ? copy(aa) : aa; bb=Base.mightalias(C,bb) ? copy(bb) : bb; W=_generic_accum(T)
    for j in 1:n, i in (upper ? (1:j) : (j:n))
        s=zero(W)
        for p in axes(aa,2)
            s=_arithmetic_madd(W(aa[i,p]),W(bb[p,j]),s)
        end
        value=W(alpha)*s; !iszero(beta) && (value+=W(beta)*W(C[i,j])); C[i,j]=convert(T,value)
    end
    C
end
function _generic_batched_execute!(Cs,As,Bs,alpha0,beta0;algorithm=:auto,kernel=:auto,threaded=true,
                                    workspace=nothing,config=nothing,batch_parallel_threshold=nothing)
    isempty(Cs) && return Cs
    T=eltype(first(Cs)); threaded=threaded && generic_threadsafe(T); cfg=config===nothing ? GemmConfig(T) : config; _generic_options(algorithm,kernel,1)
    threshold=batch_parallel_threshold===nothing ? cfg.parallel_threshold : batch_parallel_threshold
    threshold>=0 || throw(ArgumentError("negative batch threshold"))
    alpha,beta=convert(T,alpha0),convert(T,beta0)
    for i in eachindex(Cs)
        eltype(Cs[i])===eltype(As[i])===eltype(Bs[i])===T || throw(ArgumentError("batch types differ"))
        _gemm_dims(Cs[i],As[i],Bs[i])
    end
    work=sum(Float64(size(C,1))*size(C,2)*size(As[i],2) for (i,C) in enumerate(Cs))
    workers=_operation_workers(length(Cs),work,threaded,threshold)
    w=workspace===nothing ? GenericWorkspace(T;workers) : workspace
    w isa GenericWorkspace{T} || throw(ArgumentError("extended batch workspace type differs"))
    lock(w.guard) do
        any(tile->any(C->Base.mightalias(tile,C),Cs) || any(A->Base.mightalias(tile,A),As) || any(B->Base.mightalias(tile,B),Bs),w.buffers) &&
            throw(ArgumentError("batch operand aliases accumulation storage"))
        capacity=min(workers,length(w.buffers)); children=_workspace_children!(w,capacity)
        _foreach_range(length(Cs),capacity) do range,worker
            for i in range
                gemm!(Cs[i],As[i],Bs[i],alpha,beta;algorithm,kernel,threaded=false,workspace=children[worker],config=cfg)
            end
        end
    end
    Cs
end

# Generic counterparts of explicitly parallel BLAS-1 entry points. Machine
# implementations in level1_parallel.jl retain more specific dispatch. Unknown
# scalar contexts stay in the caller; generic arithmetic does not assume SIMD.
function axpy_parallel!(y::AbstractVector{T},x::AbstractVector{T},alpha=one(T);threaded=true,parallel_threshold=131072) where {T<:GenericFloat}
    parallel_threshold>=0 || throw(ArgumentError("negative parallel threshold"))
    axpy!(y,x,convert(T,alpha))
end
function scal_parallel!(x::AbstractVector{T},alpha;threaded=true,parallel_threshold=131072) where {T<:GenericFloat}
    parallel_threshold>=0 || throw(ArgumentError("negative parallel threshold"))
    _check_vector_output(x); scal!(x,convert(T,alpha))
end
function dot_parallel(x::AbstractVector{T},y::AbstractVector{T};conjugate=false,threaded=true,parallel_threshold=131072) where {T<:GenericFloat}
    parallel_threshold>=0 || throw(ArgumentError("negative parallel threshold"))
    conjugate ? dotc(x,y) : dotu(x,y)
end
function nrm2_parallel(x::AbstractVector{T};threaded=true,parallel_threshold=131072) where {T<:GenericFloat}
    parallel_threshold>=0 || throw(ArgumentError("negative parallel threshold"))
    nrm2(x)
end
function asum_parallel(x::AbstractVector{T};threaded=true,parallel_threshold=131072) where {T<:GenericFloat}
    parallel_threshold>=0 || throw(ArgumentError("negative parallel threshold"))
    asum(x)
end
