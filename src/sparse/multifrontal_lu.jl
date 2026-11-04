# SPDX-License-Identifier: Apache-2.0
struct _LUFrontUpdate{T}
    rows::Vector{Int}
    columns::Vector{Int}
    values::Matrix{T}
end
struct MultifrontalLUFactor{T,R}
    L::CSRMatrix{T,Int}
    U::CSRMatrix{T,Int}
    p::Vector{Int}
    q::Vector{Int}
    parents::Vector{Int}
    nodes::Vector{UnitRange{Int}}
    front_shapes::Vector{Tuple{Int,Int}}
    pivot_threshold::R
    growth_factor::R
end
Base.size(F::MultifrontalLUFactor)=size(F.L)

# Each original row is an element introduced at its first ordered column.
# A contribution owns its surviving rows exclusively and is sent to the first
# remaining column. Thus every entry is assembled once and all candidate rows
# for a column meet at its front, including when row pivots are off diagonal.
function _row_element_lu_tree(C,q,maxwidth)
    n=size(C,1); iq=invperm(q); original=[Int[] for _ in 1:n]; columns=[Set{Int}() for _ in 1:n]
    counts=zeros(Int,n); parents=zeros(Int,n); children=[Int[] for _ in 1:n]
    for row in 1:n
        pattern=sort!([iq[C.colind[e]] for e in C.rowptr[row]:C.rowptr[row+1]-1 if !iszero(C.values[e])])
        isempty(pattern) && throw(SingularException(1))
        firstcol=pattern[1]; push!(original[firstcol],row); counts[firstcol]+=1; union!(columns[firstcol],pattern)
    end
    for k in 1:n
        counts[k]>0 && k in columns[k] || throw(SingularException(k))
        future=sort!([j for j in columns[k] if j>k])
        if counts[k]>1
            isempty(future) && throw(SingularException(k))
            parent=future[1]; parents[k]=parent; push!(children[parent],k)
            counts[parent]+=counts[k]-1; union!(columns[parent],future)
        end
    end
    nodes=UnitRange{Int}[]; owner=zeros(Int,n); k=1
    while k<=n
        endpoint=k
        while endpoint<n && endpoint-k+1<maxwidth && parents[endpoint]==endpoint+1 && length(children[endpoint+1])==1
            endpoint+=1
        end
        push!(nodes,k:endpoint); owner[k:endpoint].=length(nodes); k=endpoint+1
    end
    group_parents=zeros(Int,length(nodes)); group_children=[Int[] for _ in nodes]; heights=ones(Int,length(nodes))
    for (g,node) in enumerate(nodes)
        parent=parents[last(node)]
        if parent!=0
            group_parents[g]=owner[parent]; push!(group_children[owner[parent]],g)
        end
        for child in group_children[g]
            heights[g]=max(heights[g],heights[child]+1)
        end
    end
    (;original,columns,counts,nodes,parents=group_parents,children=group_children,heights,iq)
end
"""Row-element multifrontal LU with separate row and column permutations.
A[p,q]=L*U. Original row elements and child Schur contributions are assembled
once, factored in fully summed panels and forwarded through an assembly tree.
Threshold partial row pivoting searches all candidate rows of the front.
Independent fronts run concurrently; inner TRSM/GEMM disable nested threading.
No dense global Schur matrix or wrapper around blocked sparse LU is used.
"""
function multifrontal_lu(A::NativeSparse;ordering=:auto,maxwidth::Union{Nothing,Integer}=nothing,
                         pivot_threshold=real(eltype(A))(0.1),threaded=true)
    n=_square(A); T=eltype(A); R=real(T)
    (maxwidth===nothing || maxwidth>=1) && isfinite(pivot_threshold) && 0<=pivot_threshold<=1 || throw(ArgumentError("invalid multifrontal LU options"))
    if ordering===:auto || ordering===:amd
        Cc=_as_csc(A); all(isfinite,Cc.values) || throw(ArgumentError("finite multifrontal LU entries required"))
        # Symmetric strategy (src/sparse/supernodal_lu.jl) for a mostly symmetric
        # pattern with a nonzero diagonal, or on request; otherwise singletons
        # then Gilbert–Peierls with AMD of AᵀA (or the supernodal engine for a
        # large core). Explicit orderings use the row-element multifrontal.
        P=_symmetric_pattern(Cc,nothing)
        (ordering===:amd || _lu_symmetric_strategy(Cc,P[1])) &&
            return _mf_lu_symmetric(Cc,real(T)(pivot_threshold),maxwidth===nothing ? typemax(Int) : Int(maxwidth),threaded,P)
        # Unsymmetric pattern: left-looking LU with an AMD(AᵀA) column order.
        return _mf_lu_unsymmetric(Cc,real(T)(pivot_threshold),maxwidth===nothing ? typemax(Int) : Int(maxwidth),threaded,P)
    end
    maxwidth=maxwidth===nothing ? 24 : maxwidth
    C=CSRMatrix(COOMatrix(CSRMatrix(A))); all(isfinite,C.values) || throw(ArgumentError("finite multifrontal LU entries required"))
    initial=maximum(abs,C.values;init=zero(R)); isfinite(initial) || throw(ArgumentError("input magnitude overflow"))
    q=sparse_ordering(C,ordering); tree=_row_element_lu_tree(C,q,Int(maxwidth)); groups=length(tree.nodes)
    updates=Vector{Union{Nothing,_LUFrontUpdate{T}}}(nothing,groups)
    lower=[Dict{Int,T}() for _ in 1:n]; upper=[Dict{Int,T}() for _ in 1:n]; p=zeros(Int,n)
    shapes=Vector{Tuple{Int,Int}}(undef,groups); growth=fill(initial,groups)
    function factor_front(g,innerthreaded)
        node=tree.nodes[g]; width=length(node); rows=Int[]
        for k in node
            append!(rows,tree.original[k])
        end
        columns=sort!(vcat(collect(node),[j for j in tree.columns[last(node)] if j>last(node)]))
        for child in tree.children[g]
            update=updates[child]; update===nothing && error("missing LU child contribution")
            append!(rows,update.rows)
        end
        sort!(rows); allunique(rows) || error("multifrontal LU row ownership collision")
        r,c=length(rows),length(columns); r>=width || throw(SingularException(first(node)))
        rowmap=Dict(row=>i for (i,row) in enumerate(rows)); colmap=Dict(col=>j for (j,col) in enumerate(columns))
        F=zeros(T,r,c); shapes[g]=(r,c)
        for k in node, original in tree.original[k], e in C.rowptr[original]:C.rowptr[original+1]-1
            value=C.values[e]; iszero(value) && continue
            F[rowmap[original],colmap[tree.iq[C.colind[e]]]]+=value
        end
        for child in tree.children[g]
            update=updates[child]
            for (j,col) in enumerate(update.columns), (i,row) in enumerate(update.rows)
                F[rowmap[row],colmap[col]]+=update.values[i,j]
            end
            updates[child]=nothing
        end
        for j in 1:width
            k=node[j]; pivot=j; largest=abs(F[j,j])
            for i in j+1:r
                value=abs(F[i,j]); value>largest && (pivot=i; largest=value)
            end
            isfinite(largest) && largest>0 || throw(SingularException(k))
            preferred=findfirst(==(q[k]),rows)
            if preferred!==nothing && preferred>=j
                magnitude=abs(F[preferred,j])
                !iszero(magnitude) && magnitude>=pivot_threshold*largest && (pivot=preferred)
            end
            if pivot!=j
                for col in 1:c
                    F[j,col],F[pivot,col]=F[pivot,col],F[j,col]
                end
                rows[j],rows[pivot]=rows[pivot],rows[j]
            end
            diagonal=F[j,j]
            for i in j+1:r
                F[i,j]/=diagonal
                for col in j+1:width
                    F[i,col]=_native_madd(-F[i,j],F[j,col],F[i,col])
                end
            end
        end
        if c>width
            trsm!(view(F,1:width,width+1:c),view(F,1:width,1:width);uplo=:L,unit=true,threaded=innerthreaded)
        end
        for j in 1:width
            k=node[j]; p[k]=rows[j]
            for i in j+1:r
                value=F[i,j]; !iszero(value) && (lower[rows[i]][k]=value)
            end
            for col in j:c
                value=F[j,col]; !iszero(value) && (upper[k][columns[col]]=value)
            end
        end
        if r>width && c>width
            # Explicit panel snapshots avoid conservative parent-view alias
            # copies inside GEMM and are private to this front/task.
            L21=copy(view(F,width+1:r,1:width)); U12=copy(view(F,1:width,width+1:c))
            gemm!(view(F,width+1:r,width+1:c),L21,U12,-one(T),one(T);threaded=innerthreaded)
            updates[g]=_LUFrontUpdate(rows[width+1:r],columns[width+1:c],copy(view(F,width+1:r,width+1:c)))
        elseif tree.parents[g]!=0
            error("inconsistent LU assembly tree")
        end
        all(isfinite,F) || throw(ArgumentError("multifrontal LU arithmetic overflow"))
        magnitude=initial
        for j in 1:c, i in 1:min(j,width)
            magnitude=max(magnitude,abs(F[i,j]))
        end
        r>width && c>width && (magnitude=max(magnitude,maximum(abs,view(F,width+1:r,width+1:c);init=zero(R))))
        growth[g]=magnitude
    end
    for height in 1:maximum(tree.heights;init=0)
        jobs=findall(==(height),tree.heights); workers=threaded ? min(length(jobs),Threads.nthreads()) : 1
        _foreach_range(length(jobs),workers) do range,_
            for i in range
                factor_front(jobs[i],threaded && workers==1)
            end
        end
    end
    sort(p)==collect(1:n) || error("multifrontal LU did not eliminate each original row once")
    lrows=[copy(lower[p[i]]) for i in 1:n]
    for i in 1:n
        lrows[i][i]=one(T)
    end
    MultifrontalLUFactor(_rows_to_csr(lrows,T,n),_rows_to_csr(upper,T,n),p,q,tree.parents,tree.nodes,shapes,
                         R(pivot_threshold),iszero(initial) ? one(R) : maximum(growth;init=initial)/initial)
end
function solve(F::MultifrontalLUFactor{T},b::AbstractVecOrMat{T};trans=:N) where T
    _trans_kind(trans)==0 || return _sparse_lu_solve(F.L,F.U,F.p,F.q,b;trans)
    n=length(F.p); Base.require_one_based_indexing(b); size(b,1)==n || throw(DimensionMismatch("sparse LU RHS differs"))
    # A[p,q]·x[q] = b[p]: row-oriented substitutions on the CSR factors.
    X=ndims(b)==1 ? b[F.p] : b[F.p,:]
    for j in axes(X,2)
        _lu_csr_solve!(view(X,:,j),F.L,F.U)
    end
    result=similar(X)
    ndims(b)==1 ? (result[F.q]=X) : (result[F.q,:]=X)
    result
end

function _direct_permutation_sign(p)
    visited=falses(length(p)); cycles=0
    for i in eachindex(p)
        visited[i] && continue
        cycles+=1; j=i
        while !visited[j]
            visited[j]=true; j=p[j]
        end
    end
    isodd(length(p)-cycles) ? -1 : 1
end
function logabsdet_factor(F::Union{MultifrontalLUFactor{T},SparseStaticLUFactor{T}}) where T
    value=zero(real(T)); phase=T(_direct_permutation_sign(F.p)*_direct_permutation_sign(F.q))
    for i in eachindex(F.p)
        d=F.U[i,i]; magnitude=abs(d); value+=log(magnitude); phase*=d/magnitude
    end
    value,phase
end
function logabsdet_factor(F::SparseLDLFactor{T}) where T
    value=zero(real(T)); phase=one(T); k=1
    for step in F.blocks
        if step==1
            d=F.diagonal[k]; magnitude=abs(d); value+=log(magnitude); phase*=d/magnitude
        else
            scale,_,_,_,_,det=_ldl_block_scaled(F.diagonal[k],F.offdiagonal[k],F.diagonal[k+1],F.hermitian,k)
            magnitude=abs(det); value+=2log(scale)+log(magnitude); phase*=det/magnitude
        end
        k+=step
    end
    value,phase
end
function determinant(F::Union{MultifrontalLUFactor,SparseStaticLUFactor,SparseLDLFactor})
    value,phase=logabsdet_factor(F); phase*exp(value)
end
