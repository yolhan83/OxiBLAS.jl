# SPDX-License-Identifier: Apache-2.0
# Layered Hopcroft–Karp with iterative DFS: no Julia call-stack depth depends
# on the length of an alternating path. Adjacency lists are canonical/sorted.
function _hopcroft_karp(adjacency,nright)
    nleft=length(adjacency); rows=zeros(Int,nleft); columns=zeros(Int,nright)
    distances=fill(-1,nleft); queue=Vector{Int}(undef,nleft)
    cursors=ones(Int,nleft); stack=Int[]; edges=Int[]
    while true
        head=1; tail=0; shortest=nleft+1
        for u in 1:nleft
            distances[u]=rows[u]==0 ? 0 : -1
            if rows[u]==0
                tail+=1; queue[tail]=u
            end
        end
        while head<=tail
            u=queue[head]; head+=1; distances[u]>=shortest && continue
            for v in adjacency[u]
                paired=columns[v]
                if paired==0
                    shortest=min(shortest,distances[u]+1)
                elseif distances[paired]<0
                    distances[paired]=distances[u]+1; tail+=1; queue[tail]=paired
                end
            end
        end
        shortest==nleft+1 && break
        fill!(cursors,1); augmented=0
        for root in 1:nleft
            rows[root]==0 && distances[root]==0 || continue
            empty!(stack); empty!(edges); push!(stack,root)
            found=false
            while !isempty(stack) && !found
                u=stack[end]; descended=false
                while cursors[u]<=length(adjacency[u])
                    v=adjacency[u][cursors[u]]; cursors[u]+=1; paired=columns[v]
                    if paired==0 && distances[u]+1==shortest
                        rows[u]=v; columns[v]=u
                        for depth in length(stack)-1:-1:1
                            previous=stack[depth]; col=edges[depth]
                            rows[previous]=col; columns[col]=previous
                        end
                        augmented+=1; found=true; break
                    elseif paired!=0 && distances[paired]==distances[u]+1 && distances[paired]<shortest
                        push!(edges,v); push!(stack,paired); descended=true; break
                    end
                end
                if !found && !descended
                    distances[u]=-1; pop!(stack)
                    !isempty(edges) && pop!(edges)
                end
            end
        end
        augmented>0 || error("inconsistent Hopcroft–Karp layering")
    end
    (;rows,columns,cardinality=count(!iszero,rows))
end
function _matrix_bipartite_matching(A::NativeSparse)
    C=CSRMatrix(A)
    adjacency=[Int[C.colind[e] for e in C.rowptr[u]:C.rowptr[u+1]-1 if !iszero(C.values[e])] for u in 1:C.m]
    _hopcroft_karp(adjacency,C.n)
end
"""Maximum-cardinality matching of a square undirected bipartite graph.
The off-diagonal pattern is symmetrized; diagonal and numerical zeros ignored.
Returns nothing for a non-bipartite graph. This differs from bipartite_matching,
which treats a rectangular matrix as a row/column bipartite incidence pattern.
"""
function bipartite_graph_matching(A::NativeSparse)
    coloring=bipartite_coloring(A); coloring.is_bipartite || return nothing
    graph=_sparse_graph(A); left=coloring.left; right=coloring.right
    indices=zeros(Int,length(graph))
    for (i,v) in enumerate(right)
        indices[v]=i
    end
    adjacency=[sort!(Int[indices[v] for v in graph[u]]) for u in left]
    matching=_hopcroft_karp(adjacency,length(right)); mate=zeros(Int,length(graph))
    edges=Tuple{Int,Int}[]
    for (i,j) in enumerate(matching.rows)
        j==0 && continue
        u,v=left[i],right[j]; mate[u]=v; mate[v]=u; push!(edges,(u,v))
    end
    (;mate,edges,cardinality=matching.cardinality,left,right,
      saturated_smaller_side=matching.cardinality==min(length(left),length(right)),
      covers_all_vertices=2matching.cardinality==length(graph))
end
