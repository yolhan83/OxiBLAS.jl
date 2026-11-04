# SPDX-License-Identifier: Apache-2.0
# All batches are prepared before the first output write. Outer parallelism
# owns one workspace per task and always disables nested GEMM parallelism.
function _batched_execute!(Cs,As,Bs,α,β;algorithm=:auto,kernel=:auto,threaded=true,
                           workspace=nothing,config=nothing,batch_parallel_threshold=nothing)
    isempty(Cs) && return Cs
    T=eltype(first(Cs)); T<:GenericFloat || throw(ArgumentError("unsupported batch element type"))
    !(T<:NativeFloat) && return _generic_batched_execute!(Cs,As,Bs,α,β;algorithm,kernel,threaded,workspace,config,batch_parallel_threshold)
    cfg=config===nothing ? GemmConfig(T) : config
    threshold=batch_parallel_threshold===nothing ? cfg.parallel_threshold : batch_parallel_threshold
    threshold>=0 || throw(ArgumentError("negative batch parallel threshold"))
    α,β=convert(T,α),convert(T,β)
    mm=nn=kk=0; work=0.0
    for i in eachindex(Cs)
        eltype(Cs[i])===eltype(As[i])===eltype(Bs[i])===T || throw(ArgumentError("batch element types must agree"))
        m,n,k=_gemm_dims(Cs[i],As[i],Bs[i])
        _l3_validate(Cs[i],algorithm,kernel,1)
        mm=max(mm,m); nn=max(nn,n); kk=max(kk,k); work+=Float64(m)*n*k
    end
    wanted=_operation_workers(length(Cs),work,threaded,threshold)
    opts=_l3_options(T,mm,nn,kk,algorithm,kernel,threaded,workspace,cfg)
    w=opts.workspace
    workers=w isa GemmWorkspace ? min(wanted,length(w.buffers)) : wanted   # captured by run()
    function run()
        if w isa GemmWorkspace
            for batch in eachindex(Cs)
                _workspace_operands_check(w,Cs[batch],As[batch],Bs[batch])
            end
        end
        children=w isa GemmWorkspace ? _workspace_children!(w,max(1,workers)) : fill(nothing,max(1,workers))
        _foreach_range(length(Cs),workers) do range,worker
            localopts=merge(opts,(;workspace=children[worker],threaded=false))
            for i in range
                gemm!(Cs[i],As[i],Bs[i],α,β;localopts...)
            end
        end
        Cs
    end
    w isa GemmWorkspace ? lock(run,w.guard) : run()
end

"""Independent batched GEMM with cross-batch alias snapshots and task-owned panels.
`threaded=false` disables the outer schedule; no nested GEMM tasks are created.
A homogeneous batch may reuse one `GemmWorkspace` sized for its largest members.
"""
function gemm_batched!(Cs,As,Bs,α=1,β=0;transA=:N,transB=:N,kwargs...)
    Base.require_one_based_indexing(Cs,As,Bs)
    _trans_kind(transA); _trans_kind(transB)
    length(Cs)==length(As)==length(Bs) || throw(DimensionMismatch("batch counts differ"))
    active=!isempty(Cs) && !iszero(convert(eltype(first(Cs)),α))
    aa=map(As) do A
        a=_op(A,transA)
        active && any(C->Base.mightalias(a,C),Cs) ? copy(a) : a
    end
    bb=map(Bs) do B
        b=_op(B,transB)
        active && any(C->Base.mightalias(b,C),Cs) ? copy(b) : b
    end
    for i in eachindex(Cs), j in 1:i-1
        Base.mightalias(Cs[i],Cs[j]) && throw(ArgumentError("batch outputs must be disjoint"))
    end
    _batched_execute!(Cs,aa,bb,α,β;kwargs...)
    Cs
end

"""GEMM on 3-D Julia arrays; dimension three is the independent batch index."""
function gemm_strided_batched!(C::AbstractArray{T,3},A::AbstractArray{T,3},B::AbstractArray{T,3},α=one(T),β=zero(T);transA=:N,transB=:N,kwargs...) where {T<:GenericFloat}
    Base.require_one_based_indexing(C,A,B)
    size(C,3)==size(A,3)==size(B,3) || throw(DimensionMismatch("batch counts differ"))
    _check_nd_output(C)
    an=_trans_kind(transA)==0; bn=_trans_kind(transB)==0
    am,ak=an ? (size(A,1),size(A,2)) : (size(A,2),size(A,1))
    bk,bncols=bn ? (size(B,1),size(B,2)) : (size(B,2),size(B,1))
    ak==bk && (size(C,1),size(C,2))==(am,bncols) || throw(DimensionMismatch("batched GEMM dimensions differ"))
    active=!iszero(convert(T,α))
    AA=active && Base.mightalias(C,A) ? copy(A) : A
    BB=active && Base.mightalias(C,B) ? copy(B) : B
    cs=[view(C,:,:,i) for i in axes(C,3)]
    aa=[_op(view(AA,:,:,i),transA) for i in axes(C,3)]
    bb=[_op(view(BB,:,:,i),transB) for i in axes(C,3)]
    _batched_execute!(cs,aa,bb,α,β;kwargs...)
    C
end

# A sufficient injectivity check for arbitrary-rank strided views, including
# negative strides. Non-strided gathers are accepted only with unique indices.
function _check_nd_output(C)
    if C isa StridedArray
        order=sortperm(collect(abs.(strides(C)))); span=1
        for d in order
            size(C,d)<=1 && continue
            abs(stride(C,d))>=span || throw(ArgumentError("overlapping output layout"))
            span=Base.checked_add(span,Base.checked_mul(size(C,d)-1,abs(stride(C,d))))
        end
    elseif C isa SubArray
        for idx in parentindices(C)
            idx isa AbstractVector && !allunique(idx) && throw(ArgumentError("output view contains repeated indices"))
        end
        _check_nd_output(parent(C))
    elseif C isa Union{Transpose,Adjoint,Base.PermutedDimsArray,Base.ReshapedArray}
        _check_nd_output(parent(C))
    elseif !(C isa Array)
        throw(ArgumentError("output must have a provably disjoint Julia array layout"))
    end
    nothing
end
