# SPDX-License-Identifier: Apache-2.0
struct _PartitionGraph
    adjacency::Vector{Dict{Int,Float64}}
    weights::Vector{Float64}
end
function _partition_graph(A;edge_weights=:unit,vertex_weights=nothing)
    n=_square(A); C=CSRMatrix(A)
    edge_weights in (:unit,:abs) || throw(ArgumentError("edge weights must be :unit or :abs"))
    all(isfinite,C.values) || throw(ArgumentError("finite partition matrix required"))
    adjacency=[Dict{Int,Float64}() for _ in 1:n]; scale=1.0
    for i in 1:n, p in C.rowptr[i]:C.rowptr[i+1]-1
        j=C.colind[p]; (i==j || iszero(C.values[p])) && continue
        w=edge_weights===:unit ? 1.0 : _matching_weight(C.values[p],:abs)
        w=max(w,get(adjacency[i],j,0.0)); adjacency[i][j]=w; adjacency[j][i]=w
    end
    if edge_weights===:abs
        scale=maximum((w for neighbors in adjacency for w in values(neighbors));init=0.0)
        iszero(scale) && (scale=1.0)
        for neighbors in adjacency, j in keys(neighbors)
            neighbors[j]/=scale
            neighbors[j]>0 || throw(ArgumentError("edge weight range underflows normalized precision"))
        end
    end
    weights=ones(Float64,n); vertex_scale=1.0
    if vertex_weights!==nothing
        Base.require_one_based_indexing(vertex_weights)
        length(vertex_weights)==n || throw(DimensionMismatch("partition vertex weight count differs"))
        weights=Float64.(vertex_weights)
        all(x->isfinite(x) && x>0,weights) || throw(ArgumentError("positive finite vertex weights required"))
        vertex_scale=maximum(weights;init=0.0); iszero(vertex_scale) && (vertex_scale=1.0)
        weights./=vertex_scale
        all(>(0),weights) || throw(ArgumentError("vertex weight range underflows normalized precision"))
    end
    _PartitionGraph(adjacency,weights),scale,vertex_scale
end
function _partition_induced(G,vertices)
    map=zeros(Int,length(G.weights))
    for (i,v) in enumerate(vertices)
        map[v]=i
    end
    adjacency=[Dict{Int,Float64}() for _ in vertices]
    for (i,v) in enumerate(vertices), j in sort!(collect(keys(G.adjacency[v])))
        map[j]!=0 && (adjacency[i][map[j]]=G.adjacency[v][j])
    end
    _PartitionGraph(adjacency,G.weights[vertices])
end
function _partition_cut(G,labels)
    sum((w for i in eachindex(labels) for (j,w) in G.adjacency[i] if i<j && labels[i]!=labels[j]);init=0.0)
end
function _coarsen_partition(G,cap)
    n=length(G.weights); map=zeros(Int,n); count=0
    for i in sortperm(1:n;by=i->(length(G.adjacency[i]),i))
        map[i]!=0 && continue
        best=0; weight=-Inf
        for j in sort!(collect(keys(G.adjacency[i])))
            map[j]==0 && G.weights[i]+G.weights[j]<=cap || continue
            w=G.adjacency[i][j]
            if w>weight
                best=j; weight=w
            end
        end
        count+=1; map[i]=count; best!=0 && (map[best]=count)
    end
    adjacency=[Dict{Int,Float64}() for _ in 1:count]; weights=zeros(Float64,count)
    for i in 1:n
        weights[map[i]]+=G.weights[i]
        for j in sort!(collect(keys(G.adjacency[i])))
            i<j && map[i]!=map[j] || continue
            ci,cj=map[i],map[j]; w=get(adjacency[ci],cj,0.0)+G.adjacency[i][j]
            adjacency[ci][cj]=w; adjacency[cj][ci]=w
        end
    end
    _PartitionGraph(adjacency,weights),map
end
function _partition_bfs(G,start)
    n=length(G.weights); seen=falses(n); order=Int[]; root=start
    while length(order)<n
        queue=Int[root]; seen[root]=true; head=1
        while head<=length(queue)
            i=queue[head]; head+=1; push!(order,i)
            neighbors=sort!([j for j in keys(G.adjacency[i]) if !seen[j]];by=j->(-G.adjacency[i][j],length(G.adjacency[j]),j))
            for j in neighbors
                if !seen[j]
                    seen[j]=true; push!(queue,j)
                end
            end
        end
        length(order)==n && break
        remaining=findall(!,seen); root=remaining[argmin([(length(G.adjacency[i]),i) for i in remaining])]
    end
    order
end
function _initial_bisection(G,fraction;minleft=1,minright=1)
    n=length(G.weights); n>=minleft+minright || throw(ArgumentError("too few vertices for bisection"))
    start=argmin([(length(G.adjacency[i]),i) for i in 1:n]); first=_partition_bfs(G,start)
    orders=(first,_partition_bfs(G,last(first))); target=fraction*sum(G.weights)
    bestlabels=ones(Int,n); bestscore=(Inf,Inf,typemax(Int))
    for order in orders
        labels=fill(2,n); load=0.0; cut=0.0
        for prefix in 1:n-minright
            i=order[prefix]
            for (j,w) in G.adjacency[i]
                cut+=labels[j]==2 ? w : -w
            end
            labels[i]=1; load+=G.weights[i]; prefix<minleft && continue
            score=(abs(load-target),cut,prefix)
            if isless(score,bestscore)
                bestscore=score; bestlabels=copy(labels)
            end
        end
    end
    bestlabels
end
function _refine_bisection!(labels,G,fraction,imbalance,passes;minleft=1,minright=1)
    total=sum(G.weights); target=fraction*total; slack=imbalance*min(fraction,1-fraction)*total
    lower,upper=target-slack,target+slack; tolerance=64eps(Float64)*max(1.0,total)
    violation(x)=max(0.0,lower-x,x-upper)
    n=length(labels)
    for _ in 1:passes
        load=sum((G.weights[i] for i in 1:n if labels[i]==1);init=0.0)
        countleft=count(==(1),labels); oldviolation=violation(load)
        gains=[sum((labels[j]==labels[i] ? -w : w for (j,w) in G.adjacency[i]);init=0.0) for i in 1:n]
        best=nothing; bestscore=(-Inf,-Inf,typemin(Int),typemin(Int))
        # Returns the updated (best, bestscore): the former closure assigned
        # both captured variables (two Core.Box).
        function consider(best,bestscore,i,j,newload,gain)
            newviolation=violation(newload)
            improving=newviolation<oldviolation-tolerance || (newviolation<=oldviolation+tolerance && gain>tolerance)
            score=(-newviolation,gain,-i,-j)
            improving && isless(bestscore,score) ? ((i,j),score) : (best,bestscore)
        end
        for i in 1:n
            allowed=labels[i]==1 ? countleft>minleft : n-countleft>minright
            allowed && ((best,bestscore)=consider(best,bestscore,i,0,load+(labels[i]==1 ? -G.weights[i] : G.weights[i]),gains[i]))
        end
        boundary=[i for i in 1:n if any(labels[j]!=labels[i] for j in keys(G.adjacency[i]))]
        for i in boundary, j in boundary
            labels[i]==1 && labels[j]==2 || continue
            gain=gains[i]+gains[j]-2*get(G.adjacency[i],j,0.0)
            best,bestscore=consider(best,bestscore,i,j,load-G.weights[i]+G.weights[j],gain)
        end
        best===nothing && break
        i,j=best; labels[i]=3-labels[i]; j!=0 && (labels[j]=3-labels[j])
    end
    labels
end
function _multilevel_bisect(G,fraction,coarse_size,refine_passes,imbalance;minleft=1,minright=1,refinement=:greedy)
    graphs=_PartitionGraph[G]; maps=Vector{Int}[]
    cap=(1+imbalance)*min(fraction,1-fraction)*sum(G.weights)
    while length(graphs[end].weights)>coarse_size
        coarse,map=_coarsen_partition(graphs[end],cap)
        length(coarse.weights)<length(graphs[end].weights) || break
        length(coarse.weights)>=2 || break
        push!(maps,map); push!(graphs,coarse)
    end
    labels=_initial_bisection(graphs[end],fraction)
    _refine_partition_dispatch!(labels,graphs[end],fraction,imbalance,refine_passes,refinement)
    for level in length(maps):-1:1
        labels=labels[maps[level]]
        _refine_partition_dispatch!(labels,graphs[level],fraction,imbalance,refine_passes,refinement;
                           minleft=level==1 ? minleft : 1,minright=level==1 ? minright : 1)
    end
    left=count(==(1),labels)
    if left<minleft || length(labels)-left<minright
        labels=_initial_bisection(G,fraction;minleft,minright)
        _refine_partition_dispatch!(labels,G,fraction,imbalance,refine_passes,refinement;minleft,minright)
    end
    labels,(;levels=length(graphs),coarse_vertices=length(graphs[end].weights))
end
function _partition_refinement_option(method)
    method in (:greedy,:fm) || throw(ArgumentError("refinement must be :greedy or :fm"))
end
function _partition_options(coarse_size,refine_passes,imbalance)
    coarse_size>=2 && refine_passes>=0 && isfinite(imbalance) && 0<=imbalance<1 || throw(ArgumentError("invalid multilevel partition options"))
end

"""
    multilevel_partition(A, k=2; coarse_size=32, refine_passes=6, imbalance=0.1, kwargs...)

Native recursive k-way multilevel partitioning. Heavy-edge contraction,
BFS initial bisection, greedy moves/pair swaps or locked best-prefix FM
(refinement=:fm) on uncoarsening. Default :greedy preserves the prior policy.
No METIS-equivalent quality or balance guarantee is claimed: balanced reports
the achieved global load criterion. The algorithms are currently sequential.
"""
function multilevel_partition(A::NativeSparse,k::Integer=2;coarse_size::Integer=32,refine_passes::Integer=6,
                               imbalance=0.1,edge_weights=:unit,vertex_weights=nothing,refinement=:greedy)
    _partition_refinement_option(refinement)
    _partition_options(coarse_size,refine_passes,imbalance)
    G,edge_scale,vertex_scale=_partition_graph(A;edge_weights,vertex_weights); n=length(G.weights)
    1<=k<=max(1,n) || throw(ArgumentError("partition count must be between 1 and the vertex count"))
    labels=zeros(Int,n); history=NamedTuple[]
    # The recursion passes itself (`self`): see _einsum_search_steps.
    function recurse(self,localgraph,vertices,base,parts)
        if parts==1
            labels[vertices].=base; return
        end
        leftparts=parts÷2; rightparts=parts-leftparts
        split,stats=_multilevel_bisect(localgraph,leftparts/parts,coarse_size,refine_passes,imbalance;minleft=leftparts,minright=rightparts,refinement)
        left=findall(==(1),split); right=findall(==(2),split)
        push!(history,merge((;vertices=length(vertices),parts,base),stats))
        self(self,_partition_induced(localgraph,left),vertices[left],base,leftparts)
        self(self,_partition_induced(localgraph,right),vertices[right],base+leftparts,rightparts)
    end
    n>0 && recurse(recurse,G,collect(1:n),1,Int(k))
    parts=[findall(==(i),labels) for i in 1:k]
    normalized_loads=[sum((G.weights[v] for v in part);init=0.0) for part in parts]
    mean=sum(normalized_loads)/k
    achieved=iszero(mean) ? 0.0 : maximum(normalized_loads)/mean-1
    cut=_partition_cut(G,labels)*edge_scale; loads=normalized_loads.*vertex_scale
    (;labels,parts,nparts=Int(k),loads,cut,balanced=achieved<=imbalance+64eps(Float64),
      achieved_imbalance=achieved,requested_imbalance=imbalance,history,
      weight_overflow=!isfinite(cut) || !all(isfinite,loads),algorithm=:heavy_edge_multilevel,refinement)
end
function _separator_cover(G,labels)
    cross=[Set(j for j in keys(G.adjacency[i]) if labels[j]!=labels[i]) for i in eachindex(labels)]
    separator=falses(length(labels))
    while true
        scores=[(length(cross[i]),length(G.adjacency[i]),-i) for i in eachindex(labels)]
        i=argmax(scores); isempty(cross[i]) && break
        separator[i]=true
        for j in cross[i]
            delete!(cross[j],i)
        end
        empty!(cross[i])
    end
    findall(separator),findall(i->labels[i]==1 && !separator[i],eachindex(labels)),
                         findall(i->labels[i]==2 && !separator[i],eachindex(labels))
end
function _partition_leaf_order(G)
    rows=Int[]; cols=Int[]
    for i in eachindex(G.weights), j in sort!(collect(keys(G.adjacency[i])))
        push!(rows,i); push!(cols,j)
    end
    n=length(G.weights)
    minimum_degree_ordering(CSRMatrix(COOMatrix(n,n,rows,cols,ones(Float64,length(rows)))))
end
"""
    multilevel_nested_dissection(A; leafsize=32, coarse_size=32, refine_passes=6, imbalance=0.1, kwargs...)

Nested dissection using multilevel bisection and a greedy vertex cover of
cross edges. Every reported separator removes all left/right cross edges.
Leaf minimum-degree ordering is exact; separators need not be minimum size.
return_tree=true exposes the separator tree for structural inspection.
"""
function multilevel_nested_dissection(A::NativeSparse;leafsize::Integer=32,coarse_size::Integer=32,
                                      refine_passes::Integer=6,imbalance=0.1,edge_weights=:unit,return_tree=false,refinement=:greedy,maxdepth::Integer=50)
    _partition_refinement_option(refinement)
    leafsize>=1 && maxdepth>=0 || throw(ArgumentError("positive leaf size and nonnegative depth required"))
    _partition_options(coarse_size,refine_passes,imbalance)
    G,_,_=_partition_graph(A;edge_weights); nodes=NamedTuple[]
    function recurse(self,localgraph,vertices,depth)
        isempty(vertices) && return Int[],0
        function leaf()
            # Local: `order` below is recurse's own variable, which this
            # closure assigned (Core.Box) before.
            local leaforder=vertices[_partition_leaf_order(localgraph)]
            push!(nodes,(;vertices=copy(vertices),left=Int[],right=Int[],separator=Int[],children=(0,0),leaf=true,depth))
            leaforder,length(nodes)
        end
        (length(vertices)<=leafsize || depth>=maxdepth) && return leaf()
        labels,_=_multilevel_bisect(localgraph,0.5,coarse_size,refine_passes,imbalance;refinement)
        separator,left,right=_separator_cover(localgraph,labels)
        (isempty(left) || isempty(right)) && return leaf()
        lp,li=self(self,_partition_induced(localgraph,left),vertices[left],depth+1)
        rp,ri=self(self,_partition_induced(localgraph,right),vertices[right],depth+1)
        sp=vertices[separator[_partition_leaf_order(_partition_induced(localgraph,separator))]]
        order=vcat(lp,rp,sp)
        push!(nodes,(;vertices=copy(vertices),left=vertices[left],right=vertices[right],separator=vertices[separator],children=(li,ri),leaf=false,depth))
        order,length(nodes)
    end
    permutation,root=recurse(recurse,G,collect(1:length(G.weights)),0)
    return_tree ? (;permutation,nodes,root) : permutation
end
