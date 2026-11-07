# SPDX-License-Identifier: Apache-2.0
"""
    multiple_minimum_degree(A)

Multiple minimum degree (Liu): every independent variable of minimum
degree is eliminated in one round before the degrees are updated, on the
quotient graph with supervariables of `minimum_degree_ordering`.
"""
multiple_minimum_degree(A::NativeSparse)=_md_quotient(A,true)
function _element_neighbors!(scratch,column,elements,incidence,active,marks,stamp)
    empty!(scratch)
    for element in incidence[column], other in elements[element]
        if other!=column && active[other] && marks[other]!=stamp
            marks[other]=stamp; push!(scratch,other)
        end
    end
    scratch
end
"""
    column_minimum_degree(A; dense_threshold=nothing, absorb=true, kwargs...)

Exact column minimum-degree on implicit row-clique elements.
Ports the ordering semantics of Rust colamd/colamd_aggressive, whose upstream
documentation explicitly disclaims real COLAMD. No numerical A'*A or initial
column-intersection graph is materialized. Overlapping elements are unioned
to compute exact degrees; worst-case work/fill is still superlinear.
dense_threshold in [0,1] ignores rows denser than that fraction of columns.
absorb removes elements contained in the new fill element; no supercolumn
merging or SuiteSparse aggressive-absorption equivalence is implied.
"""
function column_minimum_degree(A::NativeSparse;dense_threshold=nothing,absorb=true,
                               tie=dense_threshold===nothing ? :column_nnz : :index)
    dense_threshold===nothing || (isfinite(dense_threshold) && 0<=dense_threshold<=1) || throw(ArgumentError("dense row threshold must be in [0,1]"))
    tie in (:column_nnz,:index) || throw(ArgumentError("unknown column ordering tie rule"))
    C=CSRMatrix(A); m,n=size(C); elements=Set{Int}[]; incidence=[Set{Int}() for _ in 1:n]
    counts=zeros(Int,n); threshold=dense_threshold===nothing ? n : floor(Int,n*dense_threshold)
    for i in 1:m
        columns=Set(C.colind[p] for p in C.rowptr[i]:C.rowptr[i+1]-1 if !iszero(C.values[p]))
        for j in columns
            counts[j]+=1
        end
        (length(columns)<2 || length(columns)>threshold) && continue
        push!(elements,columns); id=length(elements)
        for j in columns
            push!(incidence[j],id)
        end
    end
    active=trues(n); order=Int[]; marks=zeros(Int,n); stamp=0; scratch=Int[]
    for _ in 1:n
        pivot=0; score=(typemax(Int),typemax(Int),typemax(Int))
        for j in 1:n
            active[j] || continue
            stamp+=1; _element_neighbors!(scratch,j,elements,incidence,active,marks,stamp)
            candidate=(length(scratch),tie===:column_nnz ? counts[j] : 0,j)
            if isless(candidate,score)
                pivot=j; score=candidate
            end
        end
        stamp+=1; _element_neighbors!(scratch,pivot,elements,incidence,active,marks,stamp)
        neighbors=sort!(copy(scratch)); old=collect(incidence[pivot]); active[pivot]=false
        for id in old
            for j in elements[id]
                delete!(incidence[j],id)
            end
            empty!(elements[id])
        end
        if length(neighbors)>1
            fill_element=Set(neighbors)
            if absorb
                candidates=Set{Int}()
                for j in neighbors
                    union!(candidates,incidence[j])
                end
                for id in candidates
                    if issubset(elements[id],fill_element)
                        for j in elements[id]
                            delete!(incidence[j],id)
                        end
                        empty!(elements[id])
                    end
                end
            end
            push!(elements,fill_element); id=length(elements)
            for j in neighbors
                push!(incidence[j],id)
            end
        end
        push!(order,pivot)
    end
    order
end
function _checked_sparse_permutation(permutation,n)
    Base.require_one_based_indexing(permutation)
    p=collect(Int,permutation)
    length(p)==n && sort(p)==collect(1:n) || throw(ArgumentError("invalid sparse ordering permutation"))
    p
end
"""
    sparse_ordering(A, algorithm=:minimum_degree; kwargs...)

Select a native ordering; permutation[new_index] is the old index.
`:amd` is approximate minimum degree on A+Aᵀ (Cholesky, LDLᵀ); `:amd_ata`
on AᵀA without dense rows (LU) and `:amd_qr` on AᵀA (QR), the COLAMD
substitutes of CSparse. `:minimum_degree` is the exact (slower) variant.
Rectangular matrices support :natural, :column_minimum_degree, :amd_ata and
:amd_qr.
"""
function sparse_ordering(A::NativeSparse,algorithm=:minimum_degree;kwargs...)
    algorithm isa AbstractVector && return _checked_sparse_permutation(algorithm,size(A,2))
    algorithm===:natural && return collect(1:size(A,2))
    algorithm===:rcm && return reverse_cuthill_mckee(A;kwargs...)
    algorithm===:amd && return amd_ordering(A;kind=:symmetric)
    algorithm===:amd_ata && return amd_ordering(A;kind=:ata_sparse)
    algorithm===:amd_qr && return amd_ordering(A;kind=:ata)
    algorithm===:colamd && return colamd_ordering(A)
    algorithm===:minimum_degree && return minimum_degree_ordering(A;kwargs...)
    algorithm===:multiple_minimum_degree && return multiple_minimum_degree(A;kwargs...)
    algorithm===:nested_dissection && return nested_dissection(A;kwargs...)
    algorithm===:multilevel_nested_dissection && return multilevel_nested_dissection(A;kwargs...)
    algorithm===:column_minimum_degree && return column_minimum_degree(A;kwargs...)
    throw(ArgumentError("unknown sparse ordering algorithm"))
end
"""
    ordering_metrics(A, permutation)

Exact symbolic fill count and bandwidth/profile for an explicit permutation.
This can be expensive: symbolic fill is computed, not cheaply predicted.
"""
function ordering_metrics(A::NativeSparse,permutation)
    n=_square(A); p=_checked_sparse_permutation(permutation,n); graph=_sparse_graph(A)
    pattern_nnz=n+sum(length,graph;init=0)÷2
    symbolic=symbolic_cholesky(A;ordering=p); permuted=sparse_permute(A,p)
    layout=bandwidth_profile(permuted); graph_layout=graph_bandwidth_profile(permuted)
    (;permutation=p,factor_nnz=symbolic.nnz,pattern_nnz,fill=symbolic.nnz-pattern_nnz,
      fill_ratio=pattern_nnz==0 ? 1.0 : symbolic.nnz/pattern_nnz,
      bandwidth=layout.bandwidth,profile=layout.profile,
      graph_bandwidth=graph_layout.bandwidth,graph_profile=graph_layout.profile,
      graph_average_bandwidth=graph_layout.average_bandwidth)
end
function compare_orderings(A::NativeSparse;algorithms=(:natural,:rcm,:minimum_degree,:multiple_minimum_degree,:multilevel_nested_dissection))
    _square(A)
    [merge((;algorithm),ordering_metrics(A,sparse_ordering(A,algorithm))) for algorithm in algorithms]
end
function select_sparse_ordering(A::NativeSparse;kwargs...)
    comparisons=compare_orderings(A;kwargs...); isempty(comparisons) && throw(ArgumentError("at least one ordering required"))
    scores=[(r.factor_nnz,r.bandwidth,r.profile,i) for (i,r) in enumerate(comparisons)]
    (;best=comparisons[argmin(scores)],comparisons)
end
