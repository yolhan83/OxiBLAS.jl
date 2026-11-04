# SPDX-License-Identifier: Apache-2.0

# Level 1 kernels are memory bound and short, so parallel overhead dominates:
# waking sleeping Julia threads costs tens of microseconds and wake-ups are
# staggered (Apple M5 Max, Julia 1.14, measured with 6 and 18 threads). A
# sweep of sizes x worker counts gave about one worker per 16-30 us of
# sequential time and no reliable gain at 3 MiB of traffic (about 25 us). At
# the measured ~125 GB/s of sequential traffic this is _L1_GRAIN bytes per
# worker, and below _L1_MINBYTES no task is spawned; costs below are bytes of
# equivalent traffic. Without a persistent spinning pool (as in OpenBLAS) the
# scaling stays around 1.3-2.5x. An explicit
# parallel_threshold (elements) replaces the automatic cut-off, and with it at
# least two workers run (this keeps small-vector tests on the parallel path).
const _L1_GRAIN=3<<20
const _L1_MINBYTES=4<<20
function _l1_workers(n,bytes,threaded,threshold)
    limit=min(n,Threads.nthreads())
    (threaded && limit>=2) || return 1
    if threshold===nothing
        bytes>=_L1_MINBYTES || return 1
    else
        n>=threshold || return 1
    end
    clamp(bytes÷_L1_GRAIN,2,limit)
end
# The caller computes the first range instead of only waiting, which saves one
# wake-up and keeps the part of the data it touched last in its own cache.
function _l1_foreach(f,n,workers)
    workers<=1 && return (f(1:n,1);nothing)
    tasks=Vector{Task}(undef,workers-1)
    for worker in 2:workers
        tasks[worker-1]=Threads.@spawn f(_task_range(n,$worker,workers),$worker)
    end
    f(_task_range(n,1,workers),1)
    foreach(wait,tasks)
    nothing
end

function axpy_parallel!(y::AbstractVector{T},x::AbstractVector{T},alpha=one(T);threaded=true,parallel_threshold=nothing) where {T<:NativeFloat}
    _same_length(x,y); _check_vector_output(y); a=convert(T,alpha)
    iszero(a) && return y
    workers=_l1_workers(length(y),3sizeof(T)*length(y),threaded,parallel_threshold)
    workers==1 && return axpy!(y,x,a)
    xx=Base.mightalias(x,y) && x!==y ? copy(x) : x
    _l1_foreach(length(y),workers) do range,_
        axpy!(view(y,range),view(xx,range),a)
    end
    y
end
function scal_parallel!(x::AbstractVector{T},alpha;threaded=true,parallel_threshold=nothing) where {T<:NativeFloat}
    Base.require_one_based_indexing(x); _check_vector_output(x); a=convert(T,alpha)
    # Measured: scal keeps scaling with more workers than its 2n-element
    # traffic suggests, hence the 3n cost.
    workers=_l1_workers(length(x),3sizeof(T)*length(x),threaded,parallel_threshold)
    workers==1 && return scal!(x,a)
    _l1_foreach(length(x),workers) do range,_
        _scale!(view(x,range),a)
    end
    x
end
# Partial results use the sequential kernels on views (SIMD for strided input).
function dot_parallel(x::AbstractVector{T},y::AbstractVector{T};conjugate=false,threaded=true,parallel_threshold=nothing) where {T<:NativeFloat}
    _same_length(x,y); workers=_l1_workers(length(x),2sizeof(T)*length(x),threaded,parallel_threshold)
    workers==1 && return (conjugate ? dotc(x,y) : dotu(x,y))
    partial=zeros(T,workers)
    _l1_foreach(length(x),workers) do range,worker
        xv=view(x,range);yv=view(y,range)
        partial[worker]=conjugate ? dotc(xv,yv) : dotu(xv,yv)
    end
    sum(partial)
end
# Cost of one element of the sequential nrm2 kernel, in bytes of equivalent
# traffic (measured): Float32/ComplexF32 one pass widened to Float64; Float64 a
# max pass then a scaled pass with divisions; ComplexF64 the scalar scaled loop.
_nrm2_cost(::Type{T}) where T=T===Float32 ? 2sizeof(T) : T===ComplexF32 || T===Float64 ? 4sizeof(T) : 12sizeof(T)
function nrm2_parallel(x::AbstractVector{T};threaded=true,parallel_threshold=nothing) where {T<:NativeFloat}
    Base.require_one_based_indexing(x); R=real(T)
    workers=_l1_workers(length(x),_nrm2_cost(T)*length(x),threaded,parallel_threshold)
    workers==1 && return nrm2(x)
    partial=zeros(R,workers)
    _l1_foreach(length(x),workers) do range,worker
        partial[worker]=nrm2(view(x,range))
    end
    result=zero(R)
    for value in partial
        result=hypot(result,value)
    end
    result
end
function asum_parallel(x::AbstractVector{T};threaded=true,parallel_threshold=nothing) where {T<:NativeFloat}
    Base.require_one_based_indexing(x); R=real(T)
    workers=_l1_workers(length(x),sizeof(T)*length(x),threaded,parallel_threshold)
    workers==1 && return asum(x)
    partial=zeros(R,workers)
    _l1_foreach(length(x),workers) do range,worker
        partial[worker]=asum(view(x,range))
    end
    sum(partial)
end
