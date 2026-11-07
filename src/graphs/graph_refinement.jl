# SPDX-License-Identifier: Apache-2.0
# Weighted Fiduccia–Mattheyses-style passes with locked vertices and rollback
# to the best prefix. Native Float64 graph weights, matching partition policy.
function _fm_candidate!(heap,versions,locked,weights,load,direction,lower,upper,blocked)
    empty!(blocked); candidate=nothing
    while !isempty(heap)
        item=_graph_heap_pop!(heap); _,v,version=item
        (locked[v] || versions[v]!=version) && continue
        newload=load+direction*weights[v]
        if lower<=newload<=upper
            candidate=item; break
        end
        push!(blocked,item)
    end
    for item in blocked
        _graph_heap_push!(heap,item)
    end
    candidate
end
function _fm_refine_bisection!(labels,G,fraction,imbalance,passes;minleft=1,minright=1)
    n=length(labels); total=sum(G.weights); target=fraction*total
    slack=imbalance*min(fraction,1-fraction)*total
    lower,upper=target-slack,target+slack; largest=maximum(G.weights;init=0.0)
    tolerance=64eps(Float64)*max(1.0,total)
    violation(load)=max(0.0,lower-load,load-upper)
    history=NamedTuple[]
    for pass in 1:passes
        original=copy(labels); load=sum((G.weights[v] for v in 1:n if labels[v]==1);init=0.0)
        startload=load; countleft=count(==(1),labels); cut=_partition_cut(G,labels)
        startcut=cut; startviolation=violation(load); bestcut=cut; bestviolation=startviolation; beststep=0
        # Explore one vertex beyond the initial/feasible interval; only the best
        # retained prefix may improve the original balance/cut lexicographically.
        explore_lower=min(lower,load)-largest; explore_upper=max(upper,load)+largest
        gains=[sum((labels[j]==labels[v] ? -w : w for (j,w) in G.adjacency[v]);init=0.0) for v in 1:n]
        versions=zeros(Int,n); locked=falses(n); moves=Int[]
        heaps=(Tuple{Float64,Int,Int}[],Tuple{Float64,Int,Int}[])
        blocked=Tuple{Float64,Int,Int}[]
        for v in 1:n
            _graph_heap_push!(heaps[labels[v]],(-gains[v],v,versions[v]))
        end
        cuttol=64eps(Float64)*max(1.0,abs(startcut))
        for step in 1:n
            left=countleft>minleft ? _fm_candidate!(heaps[1],versions,locked,G.weights,load,-1,explore_lower,explore_upper,blocked) : nothing
            right=n-countleft>minright ? _fm_candidate!(heaps[2],versions,locked,G.weights,load,1,explore_lower,explore_upper,blocked) : nothing
            left===nothing && right===nothing && break
            if right===nothing || (left!==nothing && isless(left,right))
                item=left; right!==nothing && _graph_heap_push!(heaps[2],right)
            else
                item=right; left!==nothing && _graph_heap_push!(heaps[1],left)
            end
            _,v,_=item; side=labels[v]; direction=side==1 ? -1 : 1
            cut-=gains[v]; load+=direction*G.weights[v]; countleft+=direction
            locked[v]=true; labels[v]=3-side; push!(moves,v)
            for j in sort!(collect(keys(G.adjacency[v])))
                locked[j] && continue
                gains[j]+=(labels[j]==side ? 2 : -2)*G.adjacency[v][j]
                versions[j]+=1
                _graph_heap_push!(heaps[labels[j]],(-gains[j],j,versions[j]))
            end
            currentviolation=violation(load)
            if currentviolation<bestviolation-tolerance ||
               (currentviolation<=bestviolation+tolerance && cut<bestcut-cuttol)
                bestviolation=currentviolation; bestcut=cut; beststep=length(moves)
            end
        end
        for step in length(moves):-1:beststep+1
            v=moves[step]; labels[v]=3-labels[v]
        end
        # Recompute after rollback rather than certifying with accumulated gains.
        actualcut=_partition_cut(G,labels)
        finalload=sum((G.weights[v] for v in 1:n if labels[v]==1);init=0.0)
        finalviolation=violation(finalload)
        improved=finalviolation<startviolation-tolerance ||
                 (finalviolation<=startviolation+tolerance && actualcut<startcut-cuttol)
        if !improved
            copyto!(labels,original); actualcut=startcut; finalload=startload
            finalviolation=startviolation; beststep=0
        end
        push!(history,(;pass,explored_moves=length(moves),retained_moves=beststep,
                       cut_before=startcut,cut_after=actualcut,violation_before=startviolation,
                       violation_after=finalviolation,load_after=finalload))
        improved || break
    end
    history
end
function _refine_partition_dispatch!(labels,G,fraction,imbalance,passes,method;kwargs...)
    if method===:fm
        _fm_refine_bisection!(labels,G,fraction,imbalance,passes;kwargs...)
    else
        _refine_bisection!(labels,G,fraction,imbalance,passes;kwargs...)
    end
    labels
end
"""
    refine_partition(A, initial; method=:fm, fraction=0.5, imbalance=0.1, passes=6, kwargs...)

Refine a supplied two-way labeling using locked best-prefix FM passes or
existing greedy moves/pair swaps. Labels are copied, never modified in place.
The retained result improves balance violation first, then cut, in working
precision; balanced reports the requested left fraction, not an optimum.
Temporary exploration can exceed the requested weight tolerance by one vertex.
"""
function refine_partition(A::NativeSparse,initial::AbstractVector{<:Integer};method=:fm,
                          fraction=0.5,imbalance=0.1,passes::Integer=6,
                          edge_weights=:unit,vertex_weights=nothing)
    Base.require_one_based_indexing(initial)
    fraction=Float64(fraction); imbalance=Float64(imbalance)
    _partition_options(2,passes,imbalance); _partition_refinement_option(method)
    isfinite(fraction) && 0<fraction<1 || throw(ArgumentError("left fraction must be in (0,1)"))
    G,edge_scale,vertex_scale=_partition_graph(A;edge_weights,vertex_weights); n=length(G.weights)
    length(initial)==n || throw(DimensionMismatch("initial partition length differs"))
    all(v->v in (1,2),initial) || throw(ArgumentError("two-way labels must be 1 or 2"))
    labels=Int.(initial)
    n>=2 && !(any(==(1),labels) && any(==(2),labels)) && throw(ArgumentError("both parts must be nonempty"))
    trace=NamedTuple[]
    if n>=2
        if method===:fm
            trace=_fm_refine_bisection!(labels,G,Float64(fraction),Float64(imbalance),Int(passes))
        else
            _refine_bisection!(labels,G,Float64(fraction),Float64(imbalance),Int(passes))
        end
    end
    loads=[sum((G.weights[v] for v in 1:n if labels[v]==part);init=0.0) for part in 1:2]
    total=sum(loads); target=fraction*total; slack=imbalance*min(fraction,1-fraction)*total
    violation=max(0.0,target-slack-loads[1],loads[1]-target-slack)
    cut=_partition_cut(G,labels)*edge_scale; actual_loads=loads.*vertex_scale
    (;labels,parts=[findall(==(part),labels) for part in 1:2],nparts=2,loads=actual_loads,cut,
      balanced=violation<=64eps(Float64)*max(1.0,total),fraction,requested_imbalance=imbalance,
      balance_violation=violation*vertex_scale,method,history=trace,
      weight_overflow=!isfinite(cut) || !all(isfinite,actual_loads))
end
