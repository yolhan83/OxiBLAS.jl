# SPDX-License-Identifier: Apache-2.0
"""
    graph_bandwidth_profile(A)

Graph convention used by upstream graph::bandwidth_profile: bandwidth is
max(abs(i-j)), profile is the sum of each row's max-column minus min-column,
including the diagonal location in the envelope, and average_bandwidth is
profile/n. Distinct from bandwidth_profile's full matrix width/lower envelope.
"""
function graph_bandwidth_profile(A::NativeSparse)
    n=_square(A); C=CSRMatrix(A); lower=upper=profile=0
    for i in 1:n
        lo=hi=i
        for e in C.rowptr[i]:C.rowptr[i+1]-1
            iszero(C.values[e]) && continue
            j=C.colind[e]; lo=min(lo,j); hi=max(hi,j)
        end
        lower=max(lower,i-lo); upper=max(upper,hi-i); profile+=hi-lo
    end
    (;lower,upper,bandwidth=max(lower,upper),profile,average_bandwidth=n==0 ? 0.0 : profile/n)
end
function _level_nested_dissection(A;leafsize::Integer=32,maxdepth::Integer=50,
                                   balance_tolerance=.2,return_tree=false)
    leafsize>=1 && maxdepth>=0 && isfinite(balance_tolerance) && 0<=balance_tolerance<1 ||
        throw(ArgumentError("invalid level-set dissection options"))
    graph=_sparse_graph(A); n=length(graph); nodes=NamedTuple[]
    G=_PartitionGraph([Dict(j=>1.0 for j in graph[i]) for i in 1:n],ones(Float64,n))
    # The recursion passes itself (`self`): a local function that calls its
    # own binding captures it in a Core.Box.
    function recurse(self,vertices,depth)
        isempty(vertices) && return Int[],0
        function leaf()
            order=vertices[_partition_leaf_order(_partition_induced(G,vertices))]
            push!(nodes,(;vertices=copy(vertices),left=Int[],right=Int[],separator=Int[],children=(0,0),
                         leaf=true,kind=:leaf,depth,achieved_imbalance=0.0,balanced=true))
            order,length(nodes)
        end
        (length(vertices)<=leafsize || depth>=maxdepth) && return leaf()
        allowed=falses(n); allowed[vertices].=true
        degree(v)=count(j->allowed[j],graph[v])
        root=vertices[argmin([(degree(v),v) for v in vertices])]
        levels=_graph_levels(graph,root,allowed); reached=vcat(levels...)
        if length(reached)<length(vertices)
            component=falses(n); component[reached].=true
            left=sort!(reached); right=[v for v in vertices if !component[v]]; separator=Int[]
            kind=:components; achieved=abs(length(left)-length(right))/length(vertices)
        else
            # Pseudo-peripheral search is confined to this induced subgraph.
            while true
                candidates=levels[end]; candidate=candidates[argmin([(degree(v),v) for v in candidates])]
                expanded=_graph_levels(graph,candidate,allowed)
                length(expanded)<=length(levels) && break
                levels=expanded
            end
            length(levels)>=3 || return leaf()
            sizes=length.(levels); prefix=cumsum(sizes); total=length(vertices)
            choices=2:length(levels)-1
            function score(level)
                leftcount=prefix[level-1]; rightcount=total-prefix[level]
                discrepancy=abs(leftcount-rightcount)/total
                (max(0.0,discrepancy-balance_tolerance),sizes[level],discrepancy,level)
            end
            cut=choices[argmin([score(level) for level in choices])]
            left=vcat(levels[1:cut-1]...); right=vcat(levels[cut+1:end]...); separator=levels[cut]
            kind=:level_separator; achieved=abs(length(left)-length(right))/length(vertices)
        end
        lp,li=self(self,left,depth+1); rp,ri=self(self,right,depth+1)
        sp=isempty(separator) ? Int[] : separator[_partition_leaf_order(_partition_induced(G,separator))]
        push!(nodes,(;vertices=copy(vertices),left=copy(left),right=copy(right),separator=copy(separator),
                     children=(li,ri),leaf=false,kind,depth,achieved_imbalance=achieved,
                     balanced=achieved<=balance_tolerance+64eps(Float64)))
        vcat(lp,rp,sp),length(nodes)
    end
    # treeroot, not root: the recursion assigns its own `root` (the BFS start),
    # which was the same boxed variable as this result.
    permutation,treeroot=recurse(recurse,collect(1:n),0)
    return_tree ? (;permutation,nodes,root=treeroot) : permutation
end
