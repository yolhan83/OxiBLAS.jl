# SPDX-License-Identifier: Apache-2.0
# Sparse successive shortest augmenting paths; no dense assignment cost matrix.
function _graph_heap_push!(heap,item)
    push!(heap,item); i=length(heap)
    while i>1
        parent=i÷2; isless(heap[i],heap[parent]) || break
        heap[i],heap[parent]=heap[parent],heap[i]; i=parent
    end
end
function _graph_heap_pop!(heap)
    result=heap[1]; tail=pop!(heap); isempty(heap) && return result
    heap[1]=tail; i=1
    while 2i<=length(heap)
        child=2i
        child<length(heap) && isless(heap[child+1],heap[child]) && (child+=1)
        isless(heap[child],heap[i]) || break
        heap[i],heap[child]=heap[child],heap[i]; i=child
    end
    result
end
function _matching_weight(value,mode)
    isfinite(value) || throw(ArgumentError("finite matching weights required"))
    if mode===:value
        value isa Real || throw(ArgumentError("complex matching needs :abs or :logabs weights"))
        weight=Float64(value)
    elseif mode===:abs
        weight=Float64(abs(value))
    else
        scale=max(abs(Float64(real(value))),abs(Float64(imag(value))))
        weight=log(scale)+log(hypot(Float64(real(value))/scale,Float64(imag(value))/scale))
    end
    isfinite(weight) || throw(ArgumentError("matching weight not representable in Float64"))
    weight
end

"""
    weighted_bipartite_matching(A; objective=:cardinality, weight=:auto)

Weighted row/column bipartite matching on the nonzero pattern of A.
objective=:cardinality first maximizes cardinality, then total weight;
:weight permits unmatched vertices to maximize weight (zero-cost ties may
increase cardinality). weight=:auto uses signed real values or complex abs;
:value, :abs and :logabs are also supported. Missing/zero entries are not edges.
Uses scaled Float64 costs, sparse residual arcs and Dijkstra potentials.
Optimality is in working precision, not an exact-arithmetic certificate.
"""
function weighted_bipartite_matching(A::NativeSparse;objective=:cardinality,weight=:auto)
    objective in (:cardinality,:weight) || throw(ArgumentError("unknown matching objective"))
    mode=weight===:auto ? (eltype(A)<:Complex ? :abs : :value) : weight
    mode in (:value,:abs,:logabs) || throw(ArgumentError("unknown matching weight transform"))
    mode===:value && !(eltype(A)<:Real) && throw(ArgumentError("complex matching needs :abs or :logabs weights"))
    C=CSRMatrix(COOMatrix(CSRMatrix(A))); m,n=size(C)
    weights=[_matching_weight(value,mode) for value in C.values]
    _weighted_assignment(C,weights,objective,mode)
end
function _weighted_assignment(C,weights,objective,mode)
    m,n=size(C)
    scale=maximum(abs,weights;init=0.0); iszero(scale) && (scale=1.0)
    source=Base.checked_add(Base.checked_add(m,n),1); sink=Base.checked_add(source,1)
    adjacency=[Int[] for _ in 1:sink]; destinations=Int[]; costs=Float64[]; capacity=Bool[]
    function edge!(u,v,cost)
        id=length(destinations)+1
        push!(destinations,v); push!(costs,cost); push!(capacity,true); push!(adjacency[u],id)
        push!(destinations,u); push!(costs,-cost); push!(capacity,false); push!(adjacency[v],id+1)
        id
    end
    for i in 1:m
        edge!(source,i,0.0)
    end
    original=Vector{Int}(undef,length(weights)); potential=zeros(Float64,sink)
    for i in 1:m, p in C.rowptr[i]:C.rowptr[i+1]-1
        j=C.colind[p]; cost=-weights[p]/scale
        original[p]=edge!(i,m+j,cost); potential[m+j]=min(potential[m+j],cost)
    end
    for j in 1:n
        edge!(m+j,sink,0.0)
    end
    potential[sink]=minimum(view(potential,m+1:m+n);init=0.0)
    distance=fill(Inf,sink); parent=zeros(Int,sink); heap=Tuple{Float64,Int}[]; augmentations=0
    while true
        fill!(distance,Inf); fill!(parent,0); empty!(heap)
        distance[source]=0.0; _graph_heap_push!(heap,(0.0,source))
        while !isempty(heap)
            current,u=_graph_heap_pop!(heap); current==distance[u] || continue
            u==sink && break
            for arc in adjacency[u]
                capacity[arc] || continue
                v=destinations[arc]; reduced=costs[arc]+potential[u]-potential[v]
                tolerance=64eps(Float64)*max(1.0,abs(costs[arc]),abs(potential[u]),abs(potential[v]))
                reduced>=-tolerance || throw(ArgumentError("matching reduced costs lost numerical feasibility"))
                candidate=current+max(0.0,reduced)
                if candidate<distance[v]
                    distance[v]=candidate; parent[v]=arc; _graph_heap_push!(heap,(candidate,v))
                end
            end
        end
        isfinite(distance[sink]) || break
        pathcost=distance[sink]-potential[source]+potential[sink]
        objective===:weight && pathcost>0 && break
        for v in eachindex(potential)
            potential[v]+=min(distance[v],distance[sink])
        end
        v=sink
        while v!=source
            arc=parent[v]; arc!=0 || error("broken matching augmenting path")
            reverse=isodd(arc) ? arc+1 : arc-1
            capacity[arc]=false; capacity[reverse]=true; v=destinations[reverse]
        end
        augmentations+=1
    end
    rows=zeros(Int,m); columns=zeros(Int,n); edges=Tuple{Int,Int}[]; total=0.0; compensation=0.0
    for i in 1:m, p in C.rowptr[i]:C.rowptr[i+1]-1
        capacity[original[p]] && continue
        j=C.colind[p]; rows[i]=j; columns[j]=i; push!(edges,(i,j))
        adjusted=weights[p]/scale-compensation; next=total+adjusted
        compensation=(next-total)-adjusted; total=next
    end
    total_weight=total*scale
    (;rows,columns,edges,cardinality=length(edges),total_weight,weight_overflow=!isfinite(total_weight),
      objective,weight_mode=mode,weight_scale=scale,augmentations,optimality=:working_precision)
end

"""
    bipartite_coloring(A)

Two-color the undirected off-diagonal pattern (diagonal entries ignored).
"""
function bipartite_coloring(A::NativeSparse)
    graph=_sparse_graph(A); colors=zeros(Int,length(graph)); bipartite=true
    for start in eachindex(graph)
        colors[start]!=0 && continue
        colors[start]=1; queue=Int[start]; head=1
        while head<=length(queue)
            i=queue[head]; head+=1
            for j in sort!(collect(graph[i]))
                if colors[j]==0
                    colors[j]=3-colors[i]; push!(queue,j)
                elseif colors[j]==colors[i]
                    bipartite=false
                end
            end
        end
    end
    (;is_bipartite=bipartite,colors,left=bipartite ? findall(==(1),colors) : Int[],
      right=bipartite ? findall(==(2),colors) : Int[])
end

"""
    weighted_graph_matching(A; objective=:weight, weight=:auto)

Weighted matching of a square bipartite adjacency graph. Returns nothing
for a non-bipartite pattern. Undirected weights use max of the two transformed
directions, explicitly handling one-triangle/asymmetric storage.
"""
function weighted_graph_matching(A::NativeSparse;objective=:weight,weight=:auto)
    objective in (:cardinality,:weight) || throw(ArgumentError("unknown matching objective"))
    mode=weight===:auto ? (eltype(A)<:Complex ? :abs : :value) : weight
    mode in (:value,:abs,:logabs) || throw(ArgumentError("unknown graph matching weight transform"))
    mode===:value && !(eltype(A)<:Real) && throw(ArgumentError("complex graph matching needs :abs or :logabs weights"))
    coloring=bipartite_coloring(A); coloring.is_bipartite || return nothing
    C=CSRMatrix(A); n=size(C,1); left=coloring.left; right=coloring.right
    il=zeros(Int,n); ir=zeros(Int,n)
    for (k,v) in enumerate(left)
        il[v]=k
    end
    for (k,v) in enumerate(right)
        ir[v]=k
    end
    pairs=Dict{Tuple{Int,Int},Float64}()
    for i in 1:n, p in C.rowptr[i]:C.rowptr[i+1]-1
        j=C.colind[p]; i==j && continue; iszero(C.values[p]) && continue
        key=il[i]!=0 ? (il[i],ir[j]) : (il[j],ir[i])
        value=_matching_weight(C.values[p],mode); pairs[key]=max(get(pairs,key,-Inf),value)
    end
    # Keep zero transformed weights as structural edges, including log(abs)=0.
    rr=Int[]; cc=Int[]; vv=Float64[]
    for key in sort!(collect(keys(pairs)))
        push!(rr,key[1]); push!(cc,key[2]); push!(vv,pairs[key])
    end
    ptr=zeros(Int,length(left)+1); ptr[1]=1
    for row in rr
        ptr[row+1]+=1
    end
    for i in 2:length(ptr)
        ptr[i]+=ptr[i-1]
    end
    B=CSRMatrix(length(left),length(right),ptr,cc,vv)
    result=_weighted_assignment(B,vv,objective,mode)
    edges=[(left[i],right[j]) for (i,j) in result.edges]
    mate=zeros(Int,n)
    for (i,j) in edges
        mate[i]=j; mate[j]=i
    end
    (;mate,edges,cardinality=result.cardinality,total_weight=result.total_weight,
      weight_overflow=result.weight_overflow,objective,weight_mode=mode,optimality=result.optimality)
end

"""
    degree_sequence(A) -> Vector{Int}

Off-diagonal undirected degrees; diagonal entries and numerical zeros ignored.
"""
degree_sequence(A::NativeSparse)=length.(_sparse_graph(A))
function is_structurally_symmetric(A::NativeSparse)
    graph=_sparse_graph(A;symmetric=false)
    all(i in graph[j] for i in eachindex(graph) for j in graph[i])
end
"""
    pseudo_peripheral_vertex(A; start=nothing) -> Int

Pseudo-peripheral vertex in the starting connected component. Empty graphs
return 0; the iteration alternates minimum-degree vertices in the farthest BFS
level until its depth stops increasing. No global-diameter guarantee.
"""
function pseudo_peripheral_vertex(A::NativeSparse;start=nothing)
    graph=_sparse_graph(A); n=length(graph); n==0 && return 0
    root=start===nothing ? argmin([(length(graph[i]),i) for i in 1:n]) : start
    root isa Integer && 1<=root<=n || throw(ArgumentError("invalid peripheral start vertex"))
    levels=_graph_levels(graph,root)
    while true
        candidates=sort(levels[end];by=i->(length(graph[i]),i)); next=candidates[1]
        expanded=_graph_levels(graph,next)
        length(expanded)<=length(levels) && return next
        root=next; levels=expanded
    end
end
