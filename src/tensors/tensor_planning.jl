# SPDX-License-Identifier: Apache-2.0
# Search metadata only: numerical elements are never inspected by the planner.
# A layout is metadata only. Unknown AbstractArray layouts stay unknown; no
# dense copy, pointer conversion or element read is performed to inspect them.
function _einsum_input_layout(A)
    if A isa StridedArray || A isa Union{Transpose,Adjoint,Base.PermutedDimsArray,Base.ReshapedArray}
        try
            return Int[stride(A,d) for d in 1:ndims(A)]
        catch error
            error isa MethodError || rethrow()
        end
    end
    nothing
end
function _einsum_operand_traffic(labels,desired,dims,layout)
    entries=prod((BigInt(dims[l]) for l in unique(labels));init=BigInt(1))
    firstaxis=findfirst(l->dims[l]>1,desired)
    # Unknown layout gets a conservative gather charge. Repeated labels use
    # the sum of their physical strides (diagonal traversal).
    step=firstaxis===nothing ? 1 : layout===nothing ? 16 : abs(sum(BigInt(layout[d]) for d in eachindex(labels) if labels[d]==desired[firstaxis]))
    entries*(1+min(BigInt(16),max(BigInt(1),BigInt(step)))) # read + packing/gather model
end
function _einsum_layout_pair(la,lb,out,dims,layouta,layoutb)
    batch=[l for l in unique(la) if l in lb && l in out]
    contracted=[l for l in unique(la) if l in lb && !(l in out)]
    left=[l for l in unique(la) if !(l in lb)];right=[l for l in unique(lb) if !(l in la)]
    # Non-GEMM cases still retain a conservative scalar traversal estimate.
    ordera=vcat(left,contracted,batch);orderb=vcat(contracted,right,batch)
    traffic=_einsum_operand_traffic(la,ordera,dims,layouta)+_einsum_operand_traffic(lb,orderb,dims,layoutb)
    natural=vcat(left,right,batch)
    fast=allunique(la) && allunique(lb) && all(l in out || l in lb for l in la) && all(l in out || l in la for l in lb)
    !fast && (natural=out) # direct reductions produce a contiguous output
    entries=prod((BigInt(dims[l]) for l in out);init=BigInt(1))
    traffic+=entries*(natural==out ? 1 : 3) # output write, possible permutation read/write
    physical=Dict{Int,Int}();stride=1
    for label in natural
        physical[label]=stride;stride=_sat_mul(stride,Int(dims[label]))   # layout model only: saturate (hypersparse)
    end
    # With a pooled intermediate, the permutation can remain a view. Model
    # its strides conservatively even when the no-workspace path materializes.
    resultlayout=all(haskey(physical,l) for l in out) ? Int[physical[l] for l in out] : nothing
    traffic,resultlayout
end
_einsum_cost_key(work,traffic,peak,model,weight)=model===:layout ? (work+weight*traffic,work,traffic,peak) : (work,traffic,peak)

"""Explain a plan's arithmetic and physical-layout model without reading data.
Traffic is a heuristic scalar-entry score, not measured bytes/time or a guarantee
that a reshape copies. Reusing a plan on other strides keeps correctness but may
change the merits of its path. No measured-speed optimality is claimed.
"""
function einsum_layout_report(plan::EinsumPlan)
    labels=copy.(plan.labels);physical=copy(plan.input_layouts);work=BigInt(0);traffic=BigInt(0)
    if plan.direct
        combined=unique(vcat(plan.labels...))
        work=prod((BigInt(plan.dimensions[l]) for l in combined);init=BigInt(1))
        desired=vcat(setdiff(combined,plan.output),plan.output)
        traffic=sum((_einsum_operand_traffic(labels,filter(in(labels),desired),plan.dimensions,layout) for (labels,layout) in zip(plan.labels,physical));init=BigInt(0))
    end
    for (i,j,out) in plan.steps
        work+=prod((BigInt(plan.dimensions[l]) for l in union(labels[i],labels[j]));init=BigInt(1))
        score,layout=_einsum_layout_pair(labels[i],labels[j],out,plan.dimensions,physical[i],physical[j]);traffic+=score
        deleteat!(labels,j);deleteat!(labels,i);push!(labels,out)
        deleteat!(physical,j);deleteat!(physical,i);push!(physical,layout)
    end
    (;estimated_work=work,estimated_traffic=traffic,cost_model=plan.cost_model,layout_weight=plan.layout_weight,
      input_strides=deepcopy(plan.input_layouts),sparse_operands=copy(plan.sparse_operands),operand_nnz=copy(plan.operand_nnz),
      peak_intermediate_bytes=plan.peak_intermediate_bytes,
      direct=plan.direct,model_only=true)
end

function _einsum_search_choices(work,live,output,dimensions,element_bytes,memory_limit,layouts)
    choices=Any[]
    for i in 1:length(work)-1,j in i+1:length(work)
        needed=Set(output)
        for k in eachindex(work)
            k in (i,j) || union!(needed,work[k])
        end
        combined=unique(vcat(work[i],work[j])); out=filter(in(needed),combined)
        bytes=_einsum_bytes(out,dimensions,element_bytes)
        peak=_sat_add(_einsum_live_bytes(live),_sat_mul(2,bytes))
        memory_limit!==nothing && peak>memory_limit && continue
        workcount=prod((BigInt(dimensions[l]) for l in combined);init=BigInt(1))
        traffic,layout=_einsum_layout_pair(work[i],work[j],out,dimensions,layouts[i],layouts[j])
        push!(choices,(i,j,out,peak,workcount,traffic,layout))
    end
    choices
end
function _einsum_search_child(state,choice,dimensions,element_bytes)
    i,j,out,pairpeak,workcount,traffic,resultlayout=choice
    work=copy(state.work);live=copy(state.live);steps=copy(state.steps);layouts=copy(state.layouts)
    deleteat!(work,j);deleteat!(work,i);push!(work,out)
    deleteat!(live,j);deleteat!(live,i);push!(live,_einsum_bytes(out,dimensions,element_bytes))
    push!(steps,(i,j,out))
    deleteat!(layouts,j);deleteat!(layouts,i);push!(layouts,resultlayout)
    (;work,live,steps,layouts,peak=max(state.peak,pairpeak),cost=state.cost+workcount,traffic=state.traffic+traffic)
end
function _einsum_search_steps(labels,output,dimensions,optimize,element_bytes,memory_limit;
                              beam_width=32,search_limit=100000,layouts=nothing,cost_model=:work,layout_weight=1)
    beam_width isa Integer && beam_width>=1 || throw(ArgumentError("positive integer beam_width required"))
    search_limit isa Integer && search_limit>=1 || throw(ArgumentError("positive integer search_limit required"))
    outputbytes=_einsum_bytes(output,dimensions,element_bytes)
    memory_limit!==nothing && outputbytes>memory_limit && throw(ArgumentError("einsum output exceeds memory_limit"))
    initial=(;work=copy.(labels),live=zeros(Int,length(labels)),steps=Tuple{Int,Int,Vector{Int}}[],peak=outputbytes,cost=BigInt(0),traffic=BigInt(0),layouts=layouts===nothing ? Any[nothing for _ in labels] : copy(layouts))
    visited=Ref(0);best=Ref{Any}(nothing)
    # The recursion passes itself (`self`): a local function that calls its
    # own binding captures it in a Core.Box.
    function visit(self,state)
        visited[]+=1
        visited[]>search_limit && throw(ArgumentError("einsum search_limit exhausted; no optimality claim; increase the budget or use greedy/beam"))
        key=state->_einsum_cost_key(state.cost,state.traffic,state.peak,cost_model,layout_weight)
        best[]!==nothing && first(key(state))>first(key(best[])) && return
        if length(state.work)==1
            if best[]===nothing || isless(key(state),key(best[]))
                best[]=state
            end
            return
        end
        for choice in _einsum_search_choices(state.work,state.live,output,dimensions,element_bytes,memory_limit,state.layouts)
            self(self,_einsum_search_child(state,choice,dimensions,element_bytes))
        end
    end
    if optimize===:optimal
        visit(visit,initial)
    else
        states=Any[initial]
        for _ in 1:length(labels)-1
            next=Any[]
            for state in states,choice in _einsum_search_choices(state.work,state.live,output,dimensions,element_bytes,memory_limit,state.layouts)
                visited[]+=1
                visited[]>search_limit && throw(ArgumentError("einsum beam search_limit exhausted; increase the budget or use greedy"))
                push!(next,_einsum_search_child(state,choice,dimensions,element_bytes))
            end
            isempty(next) && break
            sort!(next;by=state->_einsum_cost_key(state.cost,state.traffic,state.peak,cost_model,layout_weight),alg=Base.Sort.MergeSort)
            resize!(next,min(length(next),Int(beam_width)));states=next
        end
        finals=filter(state->length(state.work)==1,states)
        !isempty(finals) && (best[]=first(finals))
    end
    best[]===nothing && return Tuple{Int,Int,Vector{Int}}[],outputbytes,true
    best[].steps,best[].peak,length(labels)==1
end

"""A contraction plan executed as serial tiles; arrays remain ordinary Julia arrays.
`scratch_limit` bounds the modeled live tile/intermediate payload, excluding the
caller-owned full output, inputs, alias snapshots, pool retention and packing.
Tile kernels may use native threading; reductions across tiles are ordered.
"""
struct SlicedEinsumPlan
    base::EinsumPlan
    slice_labels::Vector{Int}
    widths::Vector{Int}
    scratch_limit::Union{Nothing,Int}
    peak_intermediate_bytes::Int
    slice_count::Int
end

"""Plan explicit named-label tiles, e.g. chunks=(i=16,k=32). A sliced reduction
accumulates partial sums; slicing output labels writes disjoint output tiles.
ASCII labels, repeated labels and unsliced broadcast ellipses are supported.
There is no dense Tensor container and no process/RSS or out-of-core guarantee.
"""
function plan_einsum_sliced(specification::AbstractString,arrays::AbstractArray...;
                            chunks, scratch_limit=nothing,optimize=:beam,beam_width=32,search_limit=100000,cost_model=:work,layout_weight=1)
    base=plan_einsum(specification,arrays...;optimize=:greedy)
    scratch_limit===nothing || (scratch_limit isa Integer && scratch_limit>=0) || throw(ArgumentError("nonnegative integer scratch_limit required"))
    T=base.scalar_type
    scratch_limit!==nothing && !isbitstype(T) && throw(ArgumentError("byte-limited slicing requires inline scalar storage; provider heap payload is unknown"))
    limit=scratch_limit===nothing ? nothing : Int(scratch_limit)
    labels=Int[];widths=Int[]
    for (key,width) in pairs(chunks)
        text=string(key)
        length(text)==1 && isascii(only(text)) && isletter(only(text)) || throw(ArgumentError("chunks keys must be named ASCII einsum labels"))
        label=Int(only(text));haskey(base.dimensions,label) || throw(ArgumentError("unknown sliced label $text"))
        label in labels && throw(ArgumentError("duplicate sliced label"))
        width isa Integer && width>=1 || throw(ArgumentError("positive integer slice width required"))
        push!(labels,label);push!(widths,Int(width))
    end
    # Canonical tile order is independent of Dict insertion order.
    order=sortperm(labels);labels=labels[order];widths=widths[order]
    dimensions=copy(base.dimensions);counts=Int[]
    for (label,width) in zip(labels,widths)
        dimensions[label]=min(width,base.dimensions[label])
        push!(counts,cld(base.dimensions[label],width))
    end
    count=any(iszero,counts) ? 0 : foldl(Base.checked_mul,counts;init=1)
    steps,peak,direct=_einsum_steps(base.labels,base.output,dimensions,optimize,Base.elsize(Vector{T}),limit;beam_width,search_limit,layouts=base.input_layouts,cost_model,layout_weight)
    tiled=EinsumPlan(base.specification,base.shapes,base.labels,base.output,base.dimensions,steps,nothing,peak,direct,T,base.memory_model,cost_model,Int(layout_weight),base.input_layouts)
    SlicedEinsumPlan(tiled,labels,widths,limit,peak,count)
end

function einsum_layout_report(plan::SlicedEinsumPlan)
    base=plan.base;dims=copy(base.dimensions)
    for (label,width) in zip(plan.slice_labels,plan.widths)
        dims[label]=min(dims[label],width)
    end
    tile=EinsumPlan(base.specification,base.shapes,base.labels,base.output,dims,base.steps,nothing,
                    plan.peak_intermediate_bytes,base.direct,base.scalar_type,base.memory_model,
                    base.cost_model,base.layout_weight,base.input_layouts)
    merge(einsum_layout_report(tile),(;slice_count=plan.slice_count,tile_model=true))
end

function _sliced_validate(C,plan,arrays)
    base=plan.base;T=base.scalar_type
    Base.require_one_based_indexing(C);_check_nd_output(C)
    eltype(C)===T || throw(ArgumentError("sliced output scalar type differs"))
    size(C)==Tuple(base.dimensions[l] for l in base.output) || throw(DimensionMismatch("sliced output shape differs"))
    length(arrays)==length(base.shapes) || throw(DimensionMismatch("sliced operand count differs"))
    for (A,shape) in zip(arrays,base.shapes)
        Base.require_one_based_indexing(A)
        eltype(A)===T && size(A)==shape || throw(DimensionMismatch("sliced operand shape/type differs"))
    end
    nothing
end
function _sliced_apply!(C,plan,arrays,alpha,beta;allocator=nothing,release=nothing,kwargs...)
    iszero(alpha) && return _scale!(C,beta)
    base=plan.base
    # Snapshot all aliases before scaling the output; snapshots are not covered
    # by scratch_limit. Pool storage aliases are rejected by the outer lease.
    inputs=map(A->_is_sparse_operand(A) ? _sparse_operand(A) : Base.mightalias(C,A) ? copy(A) : A,arrays)
    _scale!(C,beta)
    dims=copy(base.dimensions);shapes=copy(base.shapes)
    tileplan=EinsumPlan(base.specification,shapes,base.labels,base.output,dims,base.steps,plan.scratch_limit,
                        plan.peak_intermediate_bytes,base.direct,base.scalar_type,base.memory_model,base.cost_model,base.layout_weight,base.input_layouts)
    counts=Tuple(cld(base.dimensions[l],w) for (l,w) in zip(plan.slice_labels,plan.widths))
    ranges=Dict{Int,UnitRange{Int}}();views=Any[inputs...]
    for tile in CartesianIndices(counts)
        for (axis,(label,width)) in enumerate(zip(plan.slice_labels,plan.widths))
            first=(tile[axis]-1)*width+1;last=first+min(width,base.dimensions[label]-first+1)-1
            ranges[label]=first:last;dims[label]=last-first+1
        end
        for (j,(A,labels)) in enumerate(zip(inputs,base.labels))
            indices=ntuple(d->haskey(ranges,labels[d]) && size(A,d)!=1 ? ranges[labels[d]] : Colon(),ndims(A))
            views[j]=_einsum_tile(A,indices);shapes[j]=size(views[j])
        end
        output_indices=Tuple(get(ranges,label,1:base.dimensions[label]) for label in base.output)
        target=view(C,output_indices...)
        result=_einsum_compute(tileplan,Tuple(views);allocator,release,reuse_result=allocator!==nothing,output=:dense,kwargs...)
        for i in eachindex(target,result)
            target[i]+=alpha*result[i]
        end
        release!==nothing && release(result)
    end
    C
end
function einsum!(C::AbstractArray{T},plan::SlicedEinsumPlan,arrays::AbstractArray...;
                  α=one(T),β=zero(T),workspace=nothing,kwargs...) where {T<:GenericFloat}
    _sliced_validate(C,plan,arrays);alpha,beta=convert(T,α),convert(T,β)
    workspace===nothing && return _sliced_apply!(C,plan,arrays,alpha,beta;kwargs...)
    _with_einsum_workspace(workspace,arrays,C) do allocator,release
        _sliced_apply!(C,plan,arrays,alpha,beta;allocator,release,kwargs...)
    end
end
function einsum(plan::SlicedEinsumPlan,arrays::AbstractArray...;workspace=nothing,kwargs...)
    T=plan.base.scalar_type;shape=Tuple(plan.base.dimensions[l] for l in plan.base.output)
    result=Array{T}(undef,shape)
    einsum!(result,plan,arrays...;workspace,kwargs...)
end
