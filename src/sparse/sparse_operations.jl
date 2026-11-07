# SPDX-License-Identifier: Apache-2.0
function sparse_transpose(A::NativeSparse;conjugate=false)
    C=CSCMatrix(A); CSRMatrix(size(A,2),size(A,1),C.colptr,C.rowind,conjugate ? conj.(C.values) : C.values)
end
function sparse_add(A::NativeSparse,B::NativeSparse,alpha=one(eltype(A)),beta=one(eltype(A)))
    size(A)==size(B) && eltype(A)===eltype(B) || throw(DimensionMismatch("sparse sum dimensions/type differ"))
    C=COOMatrix(CSRMatrix(A)); D=COOMatrix(CSRMatrix(B)); T=eltype(A)
    a,b=convert(T,alpha),convert(T,beta)
    CSRMatrix(COOMatrix(size(A,1),size(A,2),vcat(C.rows,D.rows),vcat(C.cols,D.cols),vcat(a.*C.values,b.*D.values)))
end
function sparse_permute(A::NativeSparse,p,q=p)
    Base.require_one_based_indexing(p,q); m,n=size(A)
    sort(collect(p))==collect(1:m) && sort(collect(q))==collect(1:n) || throw(ArgumentError("invalid sparse permutation"))
    ip=invperm(p); iq=invperm(q); C=COOMatrix(CSRMatrix(A))
    CSRMatrix(COOMatrix(m,n,ip[C.rows],iq[C.cols],C.values))
end
function sparse_diagonal(values::AbstractVector{T};offset=0,m=length(values)+max(-offset,0),n=length(values)+max(offset,0)) where {T<:GenericFloat}
    Base.require_one_based_indexing(values)
    min(m,n)>=0 && length(values)==max(0,min(m+min(offset,0),n-max(offset,0))) || throw(DimensionMismatch("invalid sparse diagonal dimensions"))
    rows=collect(1:length(values)).+max(-offset,0); cols=collect(1:length(values)).+max(offset,0)
    CSRMatrix(COOMatrix(m,n,rows,cols,values))
end
function sparse_norm(A::NativeSparse,kind=:fro)
    R=real(eltype(A))
    # Entrywise norms read the stored values in place (no format conversion).
    if kind in (:fro,:F,:max)
        vals=A isa SparseMatrixCSC ? (SparseArrays.nonzeros(A),) : _mul_values(A)
        kind===:max && return maximum(v->maximum(abs,v;init=zero(R)),vals)
        return length(vals)==1 ? nrm2(vals[1]) : hypot(map(nrm2,vals)...)
    end
    if kind in (:one,1)
        if A isa Union{SparseMatrixCSC,CSCMatrix}
            cp=A isa SparseMatrixCSC ? SparseArrays.getcolptr(A) : A.colptr
            vals=A isa SparseMatrixCSC ? SparseArrays.nonzeros(A) : A.values
            mcol=zero(R)
            @inbounds for j in 1:size(A,2)
                s=zero(R)
                for q in cp[j]:cp[j+1]-1; s+=abs(vals[q]); end
                s>mcol && (mcol=s)
            end
            return mcol
        end
        C_one=A isa CSRMatrix ? A : CSRMatrix(A)
        sums=zeros(R,size(C_one,2))
        vals_one=C_one.values
        cols_one=C_one.colind
        @inbounds for p in eachindex(vals_one)
            sums[cols_one[p]]+=abs(vals_one[p])
        end
        return maximum(sums;init=zero(R))
    elseif kind in (:inf,Inf)
        if A isa CSRMatrix
            mrow=zero(R)
            @inbounds for i in 1:A.m
                s=zero(R)
                for q in A.rowptr[i]:A.rowptr[i+1]-1; s+=abs(A.values[q]); end
                s>mrow && (mrow=s)
            end
            return mrow
        elseif A isa Union{SparseMatrixCSC,CSCMatrix}
            ri=A isa SparseMatrixCSC ? SparseArrays.rowvals(A) : A.rowind
            vals=A isa SparseMatrixCSC ? SparseArrays.nonzeros(A) : A.values
            sums=zeros(R,size(A,1))
            @inbounds for p in eachindex(vals)
                sums[ri[p]]+=abs(vals[p])
            end
            return maximum(sums;init=zero(R))
        end
        C_inf=CSRMatrix(A)
        mrow_inf=zero(R)
        @inbounds for i in 1:C_inf.m
            s=zero(R)
            for q in C_inf.rowptr[i]:C_inf.rowptr[i+1]-1
                s+=abs(C_inf.values[q])
            end
            s>mrow_inf && (mrow_inf=s)
        end
        return mrow_inf
    end
    throw(ArgumentError("unsupported sparse norm"))
end

function _sparse_graph(A::NativeSparse;symmetric=true)
    n=_square(A); C=CSRMatrix(A); graph=[Set{Int}() for _ in 1:n]
    for i in 1:n, p in C.rowptr[i]:C.rowptr[i+1]-1
        j=C.colind[p]
        if i!=j && !iszero(C.values[p])
            push!(graph[i],j); symmetric && push!(graph[j],i)
        end
    end
    graph
end
function connected_components(A::NativeSparse)
    graph=_sparse_graph(A); labels=zeros(Int,length(graph)); components=Vector{Vector{Int}}()
    for start in eachindex(graph)
        labels[start]!=0 && continue
        queue=Int[start]; labels[start]=length(components)+1; first=1
        while first<=length(queue)
            i=queue[first]; first+=1
            for j in sort!(collect(graph[i]))
                if labels[j]==0
                    labels[j]=labels[start]; push!(queue,j)
                end
            end
        end
        push!(components,queue)
    end
    (;labels,components,count=length(components))
end
function _graph_levels(graph,start,allowed=trues(length(graph)))
    visited=falses(length(graph)); visited[start]=true; levels=Vector{Vector{Int}}([Int[start]])
    while true
        next=Int[]
        for i in levels[end], j in sort!(collect(graph[i]))
            if allowed[j] && !visited[j]
                visited[j]=true; push!(next,j)
            end
        end
        isempty(next) && break
        push!(levels,next)
    end
    levels
end
function level_sets(A::NativeSparse,start::Integer)
    graph=_sparse_graph(A); 1<=start<=length(graph) || throw(BoundsError(graph,start))
    _graph_levels(graph,start)
end
function reverse_cuthill_mckee(A::NativeSparse)
    graph=_sparse_graph(A); n=length(graph); seen=falses(n); order=Int[]
    while length(order)<n
        candidates=findall(!,seen); start=candidates[argmin([length(graph[i]) for i in candidates])]
        queue=Int[start]; seen[start]=true; first=1
        while first<=length(queue)
            i=queue[first]; first+=1
            neighbors=sort!([j for j in graph[i] if !seen[j]];by=j->(length(graph[j]),j))
            for j in neighbors
                if !seen[j]
                    seen[j]=true; push!(queue,j)
                end
            end
        end
        append!(order,reverse(queue))
    end
    order
end
"""Exact greedy minimum-degree ordering with symbolic clique fill."""
# Minimum degree on the quotient graph with supervariables (Liu's MMD
# without its approximations): an eliminated pivot becomes an element that
# absorbs the elements next to it; its variables with identical adjacency
# are merged into one supervariable (eliminated together, mass elimination);
# the exact external degree of each variable is recounted by marking, with
# supervariable weights. Ties go to the smallest index, through a heap with
# lazy deletion. With `multiple`, every independent variable of minimum degree
# is eliminated in one round before the degrees are updated (multiple
# elimination). The explicit elimination graph (cliques of Sets) took 46 ms
# on a 70² grid.
minimum_degree_ordering(A::NativeSparse)=_md_quotient(A,false)
function _md_quotient(A::NativeSparse,multiple::Bool)
    n=_square(A); n==0 && return Int[]
    colptr,rowind=_symmetric_pattern(_as_csc(A),nothing)
    varadj=[rowind[colptr[i]:colptr[i+1]-1] for i in 1:n]
    _md_quotient_core(n,varadj,[Int[] for _ in 1:n],Vector{Vector{Int}}(undef,n),multiple)
end
# Minimum degree on a quotient graph given by variable neighbors, adjacent
# elements and element members (`elvars` may hold initial elements beyond
# the n variables).
function _md_quotient_core(n,varadj,eladj,elvars,multiple)
    eliminated=falses(n); merged=falses(n); weight=ones(Int,n)
    members=[Int[i] for i in 1:n]
    mark=zeros(Int,n); stamp=0; touched=zeros(Int,n); round=0
    degree=Vector{Int}(undef,n)
    @inbounds for i in 1:n
        stamp+=1; mark[i]=stamp; d=0
        for j in varadj[i]
            mark[j]==stamp || (mark[j]=stamp; d+=1)
        end
        for e in eladj[i], j in elvars[e]
            mark[j]==stamp || (mark[j]=stamp; d+=1)
        end
        degree[i]=d
    end
    heap=[(degree[i],i) for i in 1:n]; _md_heapify!(heap)
    order=Int[]; sizehint!(order,n); lps=Vector{Int}[]
    @inbounds while length(order)<n
        # Pivots of this round: the minimum degree (one, or all independent
        # ones with `multiple`).
        round+=1; empty!(lps); dmin=-1; pending=Tuple{Int,Int}[]
        while !isempty(heap)
            d,p=heap[1]
            if eliminated[p] || merged[p] || degree[p]!=d
                _md_pop!(heap); continue
            end
            dmin==-1 && (dmin=d)
            d==dmin || break
            _md_pop!(heap)
            if touched[p]==round
                push!(pending,(d,p)); continue
            end
            # Eliminate p (with its supervariable members).
            append!(order,members[p]); eliminated[p]=true
            stamp+=1; mark[p]=stamp; Lp=Int[]
            for j in varadj[p]
                (eliminated[j] || merged[j] || mark[j]==stamp) && continue
                mark[j]=stamp; push!(Lp,j)
            end
            for e in eladj[p], j in elvars[e]
                (eliminated[j] || merged[j] || mark[j]==stamp) && continue
                mark[j]=stamp; push!(Lp,j)
            end
            for e in eladj[p]
                elvars[e]=Int[]    # absorbed into p
            end
            elvars[p]=Lp; lpstamp=stamp
            for i in Lp
                # Edges inside Lp are covered by the element p.
                filter!(j->!eliminated[j] && !merged[j] && mark[j]!=lpstamp,varadj[i])
                filter!(e->!isempty(elvars[e]) && e!=p,eladj[i]); push!(eladj[i],p)
                touched[i]=round
            end
            push!(lps,Lp)
            multiple || break
        end
        for x in pending
            _md_push!(heap,x)
        end
        # Supervariables among the variables reached this round, then their
        # external degrees.
        for Lp in lps
            filter!(i->!merged[i] && !eliminated[i],Lp)
            hashes=Vector{UInt}(undef,length(Lp))
            for (t,i) in enumerate(Lp)
                sort!(varadj[i]); sort!(eladj[i])
                hashes[t]=hash((varadj[i],eladj[i]))
            end
            perm=sortperm(hashes)
            for x in eachindex(perm)
                i=Lp[perm[x]]; merged[i] && continue
                y=x+1
                while y<=length(perm) && hashes[perm[y]]==hashes[perm[x]]
                    j=Lp[perm[y]]
                    if !merged[j] && varadj[j]==varadj[i] && eladj[j]==eladj[i]
                        merged[j]=true; weight[i]+=weight[j]; append!(members[i],members[j])
                    end
                    y+=1
                end
            end
            for i in Lp
                merged[i] && continue
                stamp+=1; mark[i]=stamp; d=0
                for j in varadj[i]
                    (eliminated[j] || merged[j] || mark[j]==stamp) && continue
                    mark[j]=stamp; d+=weight[j]
                end
                for e in eladj[i]
                    vs=elvars[e]
                    filter!(j->!eliminated[j] && !merged[j],vs)
                    for j in vs
                        mark[j]==stamp && continue
                        mark[j]=stamp; d+=weight[j]
                    end
                end
                if d!=degree[i]
                    degree[i]=d; _md_push!(heap,(d,i))
                end
            end
        end
    end
    order
end
function _md_heapify!(h)
    for k in length(h)÷2:-1:1
        _md_down!(h,k)
    end
    h
end
function _md_down!(h,k)
    n=length(h)
    @inbounds while true
        l=2k; l>n && break
        r=l+1; c=(r<=n && h[r]<h[l]) ? r : l
        h[c]<h[k] || break
        h[c],h[k]=h[k],h[c]; k=c
    end
end
function _md_push!(h,x)
    push!(h,x); k=length(h)
    @inbounds while k>1
        q=k÷2; h[k]<h[q] || break
        h[k],h[q]=h[q],h[k]; k=q
    end
end
function _md_pop!(h)
    x=h[1]; y=pop!(h)
    if !isempty(h)
        h[1]=y; _md_down!(h,1)
    end
    x
end
"""BFS level-separator nested dissection, with exact local minimum-degree
leaves, disconnected-component handling and an explicit recursion depth cap.
Use return_tree=true to inspect separators and achieved size balance.
"""
nested_dissection(A::NativeSparse;kwargs...)=_level_nested_dissection(A;kwargs...)
"""Symbolic Cholesky pattern and elimination forest in the chosen ordering.
By default the full pattern is symmetrized; uplo=:L/:U selects one triangle
for the upstream symbolic selected-triangle contract. Elimination tree,
children, postorder and the row structure of every factor column are
computed in O(nnz(L)) (sparse_amd.jl).
"""
function symbolic_cholesky(A::NativeSparse;ordering=:natural,uplo=nothing)
    _square(A); selected=uplo===nothing ? A : _sparse_hermitian(A,uplo)
    p=sparse_ordering(selected,ordering)
    colptr,rowind,n=_permuted_symmetric_pattern(selected,p)
    parent=_etree(colptr,rowind,n)
    columns,children=_factor_columns(colptr,rowind,parent,n)
    postorder=_postorder(parent)
    (;permutation=p,parent,children,postorder,columns,nnz=sum(length,columns;init=0))
end
function bandwidth_profile(A::NativeSparse)
    C=CSRMatrix(A); lower=upper=profile=0
    for i in 1:C.m
        first=i
        for p in C.rowptr[i]:C.rowptr[i+1]-1
            iszero(C.values[p]) && continue
            j=C.colind[p]; lower=max(lower,i-j); upper=max(upper,j-i); first=min(first,j)
        end
        profile+=i-first
    end
    (;lower,upper,bandwidth=lower+upper+1,profile)
end
"""Maximum-cardinality row/column matching by layered, iterative Hopcroft–Karp.
Returns row/column partner arrays (0 means unmatched), retaining the existing
rectangular incidence interpretation. Use bipartite_graph_matching for a
square adjacency graph interpreted through an undirected two-coloring.
"""
bipartite_matching(A::NativeSparse)=_matrix_bipartite_matching(A)
