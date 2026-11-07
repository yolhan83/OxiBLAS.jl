# SPDX-License-Identifier: Apache-2.0
# Dense tensors are ordinary Julia arrays and views, never a custom container.
# Execution recipe of one pairwise step, computed once at plan construction
# from labels and dimensions only (the plan fixes every operand size), so a
# plan is immutable and safe to share between threads. The layout of the
# arrays actually passed (views, strides) is checked at each call.
struct _PairRecipe
    i::Int                       # operands of the step (positions in the live list)
    j::Int
    ma::Symbol                   # A as stored: :N matrix, :T transposed matrix, :copy permute
    mb::Symbol
    pa::Vector{Int}              # permutations to [G1 G2 batch] for :copy (and for strided-unfriendly inputs)
    pb::Vector{Int}
    ashape::NTuple{3,Int}        # (rows, cols, batch) of the stored A, B matrices
    bshape::NTuple{3,Int}
    swap::Bool                   # C' = B'·A' (result stored as [R L batch])
    tx::Symbol                   # transA/transB of the GEMM after the swap
    ty::Symbol
    cshape::NTuple{3,Int}        # GEMM result (rows, cols, batch)
    natdims::Vector{Int}         # dimensions of the natural (GEMM) label order
    operm::Vector{Int}           # natural -> result order; empty when equal
    out::Vector{Int}             # labels of the step result
end
struct EinsumPlan
    specification::String
    shapes::Vector{Tuple}
    labels::Vector{Vector{Int}}
    output::Vector{Int}
    dimensions::Dict{Int,Int}
    steps::Vector{Tuple{Int,Int,Vector{Int}}}
    memory_limit::Union{Nothing,Int}
    peak_intermediate_bytes::Int
    direct::Bool
    scalar_type::DataType
    memory_model::Symbol
    cost_model::Symbol
    layout_weight::Int
    input_layouts::Vector{Any}
    # Sparse operands (SparseTensor, SparseArrays, OxiBLAS formats) and their
    # stored-entry counts (0 for dense ones): execution takes sparse pair
    # paths for them; the pair order itself follows the dense model.
    sparse_operands::Vector{Bool}
    operand_nnz::Vector{Int}
    # Dense pairwise recipes (nothing when the plan takes another path:
    # direct, memory-bounded, repeated/reduced labels); see _einsum_recipes.
    recipes::Union{Nothing,Vector{_PairRecipe}}
end
EinsumPlan(spec,shapes,labels,out,dims,steps,limit,peak,direct,T,model,cost,weight,layouts,sparse,nonzeros)=
    EinsumPlan(spec,shapes,labels,out,dims,steps,limit,peak,direct,T,model,cost,weight,layouts,sparse,nonzeros,
               _einsum_recipes(shapes,labels,out,dims,steps,limit,direct))

# Source compatibility for the previously exported metadata constructors.
EinsumPlan(spec,shapes,labels,out,dims,steps,limit,peak,direct,T,model)=
    EinsumPlan(spec,shapes,labels,out,dims,steps,limit,peak,direct,T,model,:work,1,Any[nothing for _ in labels])
EinsumPlan(spec,shapes,labels,out,dims,steps,limit,peak,direct,T,model,cost,weight,layouts)=
    EinsumPlan(spec,shapes,labels,out,dims,steps,limit,peak,direct,T,model,cost,weight,layouts,fill(false,length(labels)),zeros(Int,length(labels)))

"""
    EinsumWorkspace(::Type{T}=Float64)

Reusable tensor-intermediate buffers, protected for concurrent callers.
An execution leases separate buffers for simultaneously live intermediates.
Returned arrays never expose these buffers; einsum! also reuses its result
scratch. Packing/index metadata may still allocate. Retained capacity is not
bounded by a plan's live-payload memory_limit. Use empty! between campaigns.
"""
mutable struct EinsumWorkspace{T<:GenericFloat}
    buffers::Vector{Vector{T}}
    leased::Vector{Bool}
    guard::ReentrantLock
    active::Bool
end
EinsumWorkspace(::Type{T}=Float64) where {T<:GenericFloat}=EinsumWorkspace{T}(Vector{T}[],Bool[],ReentrantLock(),false)
function Base.empty!(workspace::EinsumWorkspace)
    lock(workspace.guard) do
        workspace.active && throw(ArgumentError("cannot empty an active einsum workspace"))
        empty!(workspace.buffers); empty!(workspace.leased)
    end
    workspace
end
function _einsum_acquire!(workspace::EinsumWorkspace{T},shape) where T
    count=prod(shape;init=1); slot=0; capacity=typemax(Int)
    for i in eachindex(workspace.buffers)
        workspace.leased[i] && continue
        available=length(workspace.buffers[i])
        if available>=count && available<capacity
            slot=i; capacity=available
        end
    end
    if slot==0
        slot=findfirst(!,workspace.leased)
        if slot===nothing
            push!(workspace.buffers,Vector{T}(undef,count)); push!(workspace.leased,false)
            slot=length(workspace.buffers)
        else
            resize!(workspace.buffers[slot],count)
        end
    end
    workspace.leased[slot]=true
    reshape(view(workspace.buffers[slot],1:count),shape)
end
function _einsum_release!(workspace,A)
    for i in eachindex(workspace.buffers)
        workspace.leased[i] && Base.mightalias(A,workspace.buffers[i]) && (workspace.leased[i]=false)
    end
end
function _with_einsum_workspace(f,workspace::EinsumWorkspace{T},arrays,C=nothing) where T
    lock(workspace.guard) do
        workspace.active && throw(ArgumentError("recursive einsum workspace use is unsupported"))
        all(A->eltype(A)===T,arrays) || throw(ArgumentError("einsum workspace scalar type differs"))
        for buffer in workspace.buffers
            any(A->Base.mightalias(A,buffer),arrays) && throw(ArgumentError("operand aliases einsum workspace storage"))
            C!==nothing && Base.mightalias(C,buffer) && throw(ArgumentError("output aliases einsum workspace storage"))
        end
        workspace.active=true
        try
            f(shape->_einsum_acquire!(workspace,shape),A->_einsum_release!(workspace,A))
        finally
            fill!(workspace.leased,false)
            workspace.active=false
        end
    end
end

function _einsum_tokens(s)
    tokens=Any[]; i=firstindex(s)
    while i<=lastindex(s)
        c=s[i]
        if c=='.'
            startswith(SubString(s,i),"...") || throw(ArgumentError("invalid einsum ellipsis"))
            push!(tokens,:ellipsis); i=nextind(s,i,3)
        else
            isascii(c) && isletter(c) || throw(ArgumentError("einsum labels must be ASCII letters"))
            push!(tokens,Int(c)); i=nextind(s,i)
        end
    end
    count(==(:ellipsis),tokens)<=1 || throw(ArgumentError("only one ellipsis is allowed per operand"))
    tokens
end

# Dense byte estimates saturate at typemax(Int): a hypersparse chain (three
# 1e6×1e6 operands) has dense intermediates beyond Int. The planner only
# compares these sizes and checks them against memory_limit, so a saturated
# value just means "too large" (it previously threw OverflowError).
_sat_mul(a::Int,b::Int)=(r=Base.Checked.mul_with_overflow(a,b); r[2] ? typemax(Int) : r[1])
_sat_add(a::Int,b::Int)=(r=Base.Checked.add_with_overflow(a,b); r[2] ? typemax(Int) : r[1])
function _einsum_bytes(labels,dimensions,element_bytes)
    any(l->iszero(dimensions[l]),labels) && return 0
    bytes=Int(element_bytes)
    for l in labels
        bytes=_sat_mul(bytes,Int(dimensions[l]))
    end
    bytes
end
_einsum_live_bytes(live)=foldl(_sat_add,live;init=0)
function _einsum_steps(labels,output,dimensions,optimize,element_bytes,memory_limit;beam_width=32,search_limit=100000,layouts=nothing,cost_model=:work,layout_weight=1)
    cost_model in (:work,:layout) || throw(ArgumentError("cost_model must be :work or :layout"))
    layout_weight isa Integer && 0<=layout_weight<=typemax(Int) || throw(ArgumentError("nonnegative Int layout_weight required"))
    optimize in (:beam,:optimal) && return _einsum_search_steps(labels,output,dimensions,optimize,element_bytes,memory_limit;beam_width,search_limit,layouts,cost_model,layout_weight)
    optimize in (:none,:greedy,:left) || throw(ArgumentError("optimize must be :none, :greedy, :left, :beam or :optimal"))
    output_bytes=_einsum_bytes(output,dimensions,element_bytes)
    memory_limit!==nothing && output_bytes>memory_limit && throw(ArgumentError("einsum output exceeds memory_limit"))
    optimize===:none && return Tuple{Int,Int,Vector{Int}}[],output_bytes,true
    work=copy.(labels); steps=Tuple{Int,Int,Vector{Int}}[]; live=zeros(Int,length(work)); peak=output_bytes
    physical=layouts===nothing ? Any[nothing for _ in labels] : copy(layouts)
    while length(work)>1
        best=nothing; bestcost=nothing; bestlayout=nothing
        for i in 1:length(work)-1, j in i+1:length(work)
            optimize===:left && (i!=1 || j!=2) && continue
            needed=Set(output)
            for k in eachindex(work)
                k in (i,j) || union!(needed,work[k])
            end
            combined=unique(vcat(work[i],work[j]))
            out=filter(in(needed),combined)
            bytes=_einsum_bytes(out,dimensions,element_bytes)
            # A permutation may coexist with the GEMM result. Operand buffers
            # remain live until the pair is complete; original inputs excluded.
            pair_peak=_sat_add(_einsum_live_bytes(live),_sat_mul(2,bytes))
            memory_limit!==nothing && pair_peak>memory_limit && continue
            workcount=prod((BigInt(dimensions[l]) for l in combined);init=BigInt(1))
            traffic,resultlayout=_einsum_layout_pair(work[i],work[j],out,dimensions,physical[i],physical[j])
            cost=_einsum_cost_key(workcount,traffic,pair_peak,cost_model,layout_weight)
            if best===nothing || isless(cost,bestcost)
                best=(i,j,out); bestcost=cost; bestlayout=resultlayout
            end
        end
        # Direct all-operand contraction requires only the output tensor payload.
        best===nothing && return Tuple{Int,Int,Vector{Int}}[],output_bytes,true
        i,j,out=best
        bytes=_einsum_bytes(out,dimensions,element_bytes)
        peak=max(peak,_sat_add(_einsum_live_bytes(live),_sat_mul(2,bytes)))
        deleteat!(live,j); deleteat!(live,i); push!(live,bytes)
        push!(steps,(i,j,out)); deleteat!(work,j); deleteat!(work,i); push!(work,out)
        deleteat!(physical,j);deleteat!(physical,i);push!(physical,bestlayout)
    end
    steps,peak,length(labels)==1
end

"""
    plan_einsum(specification, arrays...; optimize=:greedy, memory_limit=nothing, kwargs...) -> EinsumPlan

Build a reusable shape-checked contraction plan. Supports ellipsis broadcasting,
repeated labels (diagonals/traces), scalar outputs, reductions, and any operand count.
Greedy planning minimizes local pair work, then intermediate size. `:beam`
keeps bounded-width candidate paths; `:optimal` exhaustively minimizes modeled
pair work with peak payload as a tie-breaker. Both enforce `search_limit` and
throw on exhaustion; optimality concerns this cost model, not measured runtime.
`:none` bypasses pair planning and evaluates the all-operand contraction directly.
`memory_limit` bounds live intermediate tensor payload in bytes, including the
result and a conservative output permutation copy, excluding caller inputs,
planner metadata, per-worker index vectors, and BLAS packing workspaces. It is
not a process RSS cap. If no pair fits, contract all operands directly with no
pair intermediates; this can be substantially slower. The output must fit.
"""
function plan_einsum(specification::AbstractString,arrays::AbstractArray...;optimize=:greedy,memory_limit=nothing,beam_width=32,search_limit=100000,cost_model=:work,layout_weight=1)
    isempty(arrays) && throw(ArgumentError("einsum requires at least one operand"))
    T=eltype(first(arrays)); T<:GenericFloat || throw(ArgumentError("unsupported einsum element type"))
    all(A->eltype(A)===T,arrays) || throw(ArgumentError("einsum element types must agree"))
    memory_limit===nothing || (memory_limit isa Integer && memory_limit>=0) || throw(ArgumentError("memory_limit must be a nonnegative integer byte count"))
    memory_limit!==nothing && !isbitstype(T) && throw(ArgumentError("byte-limited einsum requires inline numeric storage; reference-backed scalar payload is provider-owned"))
    limit=memory_limit===nothing ? nothing : Int(memory_limit)
    spec=filter(!isspace,String(specification)); parts=split(spec,"->")
    length(parts)<=2 || throw(ArgumentError("invalid einsum arrow"))
    inputs=split(parts[1],','); length(inputs)==length(arrays) || throw(DimensionMismatch("einsum operand count differs"))
    tokens=_einsum_tokens.(inputs); ranks=Int[]
    for (ts,A) in zip(tokens,arrays)
        Base.require_one_based_indexing(A)
        explicit=count(!=(:ellipsis),ts); ell=ndims(A)-explicit
        ell>=0 && (:ellipsis in ts || ell==0) || throw(DimensionMismatch("einsum operand rank differs"))
        push!(ranks,ell)
    end
    maxell=maximum(ranks;init=0); elllabels=collect(257:256+maxell)
    labels=Vector{Int}[]; dims=Dict{Int,Int}(); counts=Dict{Int,Int}()
    for (ts,A,rank) in zip(tokens,arrays,ranks)
        ls=Int[]
        for t in ts
            t===:ellipsis ? append!(ls,elllabels[maxell-rank+1:maxell]) : push!(ls,t)
        end
        localdims=Dict{Int,Int}()
        for (d,l) in enumerate(ls)
            s=size(A,d)
            haskey(localdims,l) && localdims[l]!=s && throw(DimensionMismatch("diagonal dimensions differ"))
            localdims[l]=s
            if haskey(dims,l) && dims[l]!=s
                l>256 && (s==1 || dims[l]==1) || throw(DimensionMismatch("einsum label dimensions differ"))
                dims[l]=s==1 ? dims[l] : s
            else
                dims[l]=s
            end
            counts[l]=get(counts,l,0)+1
        end
        push!(labels,ls)
    end
    output=Int[]
    if length(parts)==2
        for t in _einsum_tokens(parts[2])
            t===:ellipsis ? append!(output,elllabels) : push!(output,t)
        end
    else
        append!(output,elllabels)
        append!(output,sort([l for (l,c) in counts if l<=256 && c==1]))
    end
    allunique(output) || throw(ArgumentError("output labels must be unique"))
    all(haskey(dims,l) for l in output) || throw(ArgumentError("unknown einsum output label"))
    layouts=Any[_einsum_input_layout(A) for A in arrays]
    steps,peak,direct=_einsum_steps(labels,output,dims,optimize,Base.elsize(Vector{T}),limit;beam_width,search_limit,layouts,cost_model,layout_weight)
    sparse=Bool[_is_sparse_operand(A) for A in arrays]
    # Stored-entry counts without converting the operand (an O(nnz) copy
    # discarded at once, which dominated planning on sparse inputs).
    nonzeros=Int[_is_sparse_operand(A) ? _operand_nnz(A) : 0 for A in arrays]
    EinsumPlan(spec,Tuple[size(A) for A in arrays],labels,output,dims,steps,limit,peak,direct,T,isbitstype(T) ? :inline : :reference_slots,cost_model,Int(layout_weight),layouts,sparse,nonzeros)
end

# Dynamic worker calibration for tensor operations:
# Scales smoothly with problem volume and user machine threads (Threads.nthreads()).
# For single-batch contractions (batches == 1), lowering parallel_threshold to 500k
# allows gemm!'s internal _gemm_workers to scale gradually (2 workers at ~5M, 5 at ~16M,
# 14 at ~85M, etc.) instead of gating at the default 8M dense matrix threshold.
const _TENSOR_GEMM_PARALLEL_THRESHOLD = 500_000

@inline function _tensor_direct_workers(work::Real, n::Int, threaded::Bool)
    (threaded && Threads.nthreads() >= 2 && n >= 2) || return 1
    work < 80_000 && return 1
    clamp(round(Int, work / 60_000), 2, min(n, Threads.nthreads()))
end

@inline function _tensor_batched_workers(::Type{T}, batches::Int, m::Int, n::Int, k::Int, threaded::Bool) where T
    (threaded && Threads.nthreads() >= 2 && batches >= 2) || return 1
    slice_work = Float64(m) * n * k
    total_work = slice_work * batches
    (total_work < 500_000 || slice_work < 20_000) && return 1
    clamp(round(Int, total_work / 600_000), 2, min(batches, Threads.nthreads()))
end

function _einsum_direct(arrays,labels,output,dims,::Type{T};threaded=true,allocator=nothing) where T
    shape=Tuple(dims[l] for l in output); result=allocator===nothing ? Array{T}(undef,shape) : allocator(shape)
    reduced=setdiff(unique(vcat(labels...)),output)
    rshape=Tuple(dims[l] for l in reduced); rindices=CartesianIndices(rshape)
    outindices=CartesianIndices(shape)
    work=Float64(length(result))*length(rindices)*length(arrays)
    workers=_tensor_direct_workers(work,length(result),threaded)
    _foreach_range(length(result),workers) do range,_
        assignment=zeros(Int,maximum(vcat(output,reduced);init=0))
        for linear in range
            oi=outindices[linear]
            for d in eachindex(output)
                assignment[output[d]]=oi[d]
            end
            s=zero(T)
            for ri in rindices
                for d in eachindex(reduced)
                    assignment[reduced[d]]=ri[d]
                end
                v=one(T)
                for (A,ls) in zip(arrays,labels)
                    idx=ntuple(d->size(A,d)==1 ? 1 : assignment[ls[d]],ndims(A))
                    v*=A[idx...]
                end
                s+=v
            end
            result[oi]=s
        end
    end
    result
end

# Matrix layout of one pairwise-contraction operand X (labels lx) for the
# groups (G1, G2, batch): :N when lx is already [G1 G2 batch], so a plain
# reshape is a strided matrix; :T when lx is [G2 G1 batch], so the reshape is
# used transposed; :copy otherwise.
function _einsum_layout(lx,G1,G2,batch)
    lx==vcat(G1,G2,batch) && return :N
    lx==vcat(G2,G1,batch) && return :T
    :copy
end
function _einsum_operand(X,lx,G1,G2,batch,dims,mode,copies::Bool)
    r=prod(dims[l] for l in G1;init=1); c=prod(dims[l] for l in G2;init=1); b=prod(dims[l] for l in batch;init=1)
    mode===:N && return reshape(X,r,c,b),:N
    mode===:T && return reshape(X,c,r,b),:T
    perm=Tuple(findfirst(==(l),lx)::Int for l in vcat(G1,G2,batch))
    # One contiguous copy (TTGT) makes the GEMM packing read a strided array;
    # packing through a reshaped PermutedDimsArray goes element by element
    # through index divisions. Memory-bounded plans keep the lazy view.
    copies ? (reshape(permutedims(X,perm),r,c,b),:N) : (reshape(PermutedDimsArray(X,perm),r,c,b),:N)
end
_flip(t::Symbol)=t===:N ? :T : :N

# free_output: `out` is a set (an intermediate of the plan). The result keeps
# the natural GEMM order, which is returned with it: (C, labels).
function _einsum_pair(A,B,la,lb,out,dims;threaded=true,allocator=nothing,copies::Bool=true,free_output::Bool=false,kwargs...)
    T=eltype(A)
    uniqueinputs=allunique(la) && allunique(lb)
    normaldims=all(size(A,d)==dims[l] for (d,l) in enumerate(la)) && all(size(B,d)==dims[l] for (d,l) in enumerate(lb))
    # Exclusive reduced labels need a unary reduction and use the general path.
    allneeded=all(l in out || l in lb for l in la) && all(l in out || l in la for l in lb)
    if !(uniqueinputs && normaldims && allneeded)
        C=_einsum_direct((A,B),(la,lb),out,dims,T;threaded,allocator)
        return free_output ? (C,out) : C
    end
    batch=[l for l in out if l in la && l in lb]
    K1=[l for l in la if l in lb && !(l in out)]; K2=[l for l in lb if l in K1]
    L1=[l for l in la if !(l in lb)]; L2=[l for l in out if l in L1]
    R1=[l for l in lb if !(l in la)]; R2=[l for l in out if l in R1]
    # Group orders and the A/B swap (Cᵀ = Bᵀ·Aᵀ) are chosen to minimize the
    # elements copied: inputs that are not already a (transposed) matrix,
    # then the final permutation of the result.
    lenA,lenB=length(A),length(B); lenC=prod(dims[l] for l in out;init=1)
    best=nothing; bestcost=typemax(Int)
    for K in unique((K1,K2)), L in unique((L1,L2)), R in unique((R1,R2)), swap in (false,true)
        ma=_einsum_layout(la,L,K,batch); mb=_einsum_layout(lb,K,R,batch)
        cand=swap ? vcat(R,L,batch) : vcat(L,R,batch)   # not `natural`: captured below
        cost=(ma===:copy ? lenA : 0)+(mb===:copy ? lenB : 0)+(free_output || cand==out ? 0 : lenC)
        if cost<bestcost
            best=(K,L,R,swap,ma,mb,cand); bestcost=cost
        end
    end
    K,L,R,swap,ma,mb,natural=best
    m=prod(dims[l] for l in L;init=1); n=prod(dims[l] for l in R;init=1); batches=prod(dims[l] for l in batch;init=1)
    aa,ta=_einsum_operand(A,la,L,K,batch,dims,ma,copies)
    bb,tb=_einsum_operand(B,lb,K,R,batch,dims,mb,copies)
    shape=swap ? (n,m,batches) : (m,n,batches)
    c=allocator===nothing ? Array{T}(undef,shape) : allocator(shape)
    X,Y,tx,ty=swap ? (bb,aa,_flip(tb),_flip(ta)) : (aa,bb,ta,tb)
    _pair_gemm!(c,X,Y,tx,ty,one(T),zero(T),threaded;kwargs...)
    reshaped=reshape(c,Tuple(dims[l] for l in natural))
    free_output && return (reshaped,natural)
    natural==out && return reshaped
    perm=Tuple(findfirst(==(l),natural)::Int for l in out)
    allocator===nothing ? permutedims(reshaped,perm) : PermutedDimsArray(reshaped,perm)
end

# ---- recipes: the decisions of _einsum_pair, made once per plan ----------------
# Same choice as _einsum_pair (group orders and swap minimizing the copied
# elements), with the result order fixed for the last step (the plan output)
# and free for intermediate steps (natural GEMM order).
function _pair_recipe(i,j,la,lb,sa,sb,out,dims,free::Bool)
    (allunique(la) && allunique(lb)) || return nothing
    (length(sa)==length(la) && all(sa[d]==dims[l] for (d,l) in enumerate(la))) || return nothing
    (length(sb)==length(lb) && all(sb[d]==dims[l] for (d,l) in enumerate(lb))) || return nothing
    (all(l in out || l in lb for l in la) && all(l in out || l in la for l in lb)) || return nothing
    batch=[l for l in out if l in la && l in lb]
    K1=[l for l in la if l in lb && !(l in out)]; K2=[l for l in lb if l in K1]
    L1=[l for l in la if !(l in lb)]; L2=[l for l in out if l in L1]
    R1=[l for l in lb if !(l in la)]; R2=[l for l in out if l in R1]
    lenA=prod(sa;init=1); lenB=prod(sb;init=1); lenC=prod(dims[l] for l in out;init=1)
    best=nothing; bestcost=typemax(Int)
    for K in unique((K1,K2)), L in unique((L1,L2)), R in unique((R1,R2)), swap in (false,true)
        ma=_einsum_layout(la,L,K,batch); mb=_einsum_layout(lb,K,R,batch)
        cand=swap ? vcat(R,L,batch) : vcat(L,R,batch)   # not `natural`: captured below
        cost=(ma===:copy ? lenA : 0)+(mb===:copy ? lenB : 0)+(free || cand==out ? 0 : lenC)
        cost<bestcost && (best=(K,L,R,swap,ma,mb,cand); bestcost=cost)
    end
    K,L,R,swap,ma,mb,natural=best
    p(G)=prod(dims[l] for l in G;init=1)
    m,k,n,b=p(L),p(K),p(R),p(batch)
    pa=ma===:copy ? [findfirst(==(l),la)::Int for l in vcat(L,K,batch)] : Int[]
    pb=mb===:copy ? [findfirst(==(l),lb)::Int for l in vcat(K,R,batch)] : Int[]
    ashape=ma===:T ? (k,m,b) : (m,k,b); bshape=mb===:T ? (n,k,b) : (k,n,b)
    ta=ma===:T ? :T : :N; tb=mb===:T ? :T : :N
    tx,ty=swap ? (_flip(tb),_flip(ta)) : (ta,tb)
    result=free ? natural : out
    operm=natural==result ? Int[] : [findfirst(==(l),natural)::Int for l in result]
    _PairRecipe(i,j,ma,mb,pa,pb,ashape,bshape,swap,tx,ty,swap ? (n,m,b) : (m,n,b),[dims[l] for l in natural],operm,result)
end
function _einsum_recipes(shapes,labels,output,dims,steps,limit,direct)
    (direct || limit!==nothing || isempty(steps) || !allunique(output)) && return nothing
    ls=copy(labels); ss=Any[s for s in shapes]; recipes=_PairRecipe[]
    for (step,(i,j,out)) in enumerate(steps)
        last=step==length(steps)
        if last
            (length(out)==length(output) && issetequal(out,output)) || return nothing
            out=output
        else
            allunique(out) || return nothing
        end
        r=_pair_recipe(i,j,ls[i],ls[j],ss[i],ss[j],out,dims,!last)
        r===nothing && return nothing
        push!(recipes,r)
        deleteat!(ls,j); deleteat!(ls,i); push!(ls,r.out)
        deleteat!(ss,j); deleteat!(ss,i); push!(ss,Tuple(dims[l] for l in r.out))
    end
    recipes
end

# ---- in-place executor -----------------------------------------------------------
# Column-major contiguity (size-1 dimensions ignored): any grouping reshape is
# then a matrix with unit stride and leading dimension = rows, which is the
# layout the GEMM pointer path accepts. Being a StridedArray is not enough
# (a strided view may have a non-unit first stride or gaps).
function _colmajor(X)
    X isa StridedArray || return false
    s=1
    for d in 1:ndims(X)
        size(X,d)==1 || stride(X,d)==s || return false
        s*=size(X,d)
    end
    true
end
_ws_tmp(::Nothing,::Type{T},dims) where T=Array{T}(undef,dims)
_ws_tmp(ws::EinsumWorkspace,::Type{T},dims) where T=_einsum_acquire!(ws,dims)
_ws_free(::Nothing,X)=nothing
_ws_free(ws::EinsumWorkspace,X)=(X===nothing || _einsum_release!(ws,X); nothing)
# Permuted copy dst = permutedims(src, perm) for a column-major contiguous src
# and a contiguous dst, specialized on the rank (no PermutedDimsArray of
# runtime type, no allocation); other ranks or layouts use Base.permutedims!.
# src is read through its column-major linear index with dimensions sdims
# (src's own order); dst is written linearly. Both must be contiguous, which
# the callers guarantee (workspace or Array buffers, _colmajor inputs).
function _permute_copy!(dst,src,perm::Vector{Int},sdims=size(src))
    N=length(perm)
    N==2 && return _permute_copy!(dst,src,perm,sdims,Val(2))
    N==3 && return _permute_copy!(dst,src,perm,sdims,Val(3))
    N==4 && return _permute_copy!(dst,src,perm,sdims,Val(4))
    N==5 && return _permute_copy!(dst,src,perm,sdims,Val(5))
    permutedims!(reshape(dst,Tuple(sdims[d] for d in perm)),reshape(src,Tuple(sdims)),Tuple(perm))
end
function _permute_copy!(dst,src,perm::Vector{Int},sdims,::Val{N}) where N
    sz=ntuple(k->sdims[perm[k]],Val(N))
    st=ntuple(Val(N)) do k
        s=1
        for d in 1:perm[k]-1; s*=sdims[d]; end
        s
    end
    I=0
    @inbounds for c in CartesianIndices(sz)
        off=1
        for k in 1:N; off+=(c[k]-1)*st[k]; end
        I+=1; dst[I]=src[off]
    end
    dst
end
# Operand as a GEMM matrix. Necessary TTGT copies: a :copy layout (permuted
# into contiguous storage), or an input whose layout the kernel would not
# accept (copied as is). With a workspace they come from its pool. Each
# branch calls on with concrete types (no Union result, so no boxing).
function _pair_exec!(D,A,B,r::_PairRecipe,α,β,ws,::Type{T},threaded::Bool) where T
    # Temporaries are flat (length known, rank-independent type) and then
    # reshaped to the recipe's 3-D matrix form: concrete types throughout.
    if r.ma!==:copy && _colmajor(A)
        _pair_exec_b!(D,reshape(A,r.ashape),r.ma,nothing,B,r,α,β,ws,T,threaded)
    elseif r.ma===:copy && _colmajor(A)
        buf=_ws_tmp(ws,T,(length(A),)); _permute_copy!(buf,A,r.pa)
        _pair_exec_b!(D,reshape(buf,r.ashape),:N,buf,B,r,α,β,ws,T,threaded)
    else   # layout the kernel would not accept: contiguous copy first
        buf=_ws_tmp(ws,T,(length(A),)); copyto!(buf,A)
        if r.ma===:copy
            buf2=_ws_tmp(ws,T,(length(A),)); _permute_copy!(buf2,buf,r.pa,size(A)); _ws_free(ws,buf)
            _pair_exec_b!(D,reshape(buf2,r.ashape),:N,buf2,B,r,α,β,ws,T,threaded)
        else
            _pair_exec_b!(D,reshape(buf,r.ashape),r.ma,buf,B,r,α,β,ws,T,threaded)
        end
    end
end
function _pair_exec_b!(D,X,ta,bufA,B,r::_PairRecipe,α,β,ws,::Type{T},threaded::Bool) where T
    if r.mb!==:copy && _colmajor(B)
        _pair_core!(D,X,ta,reshape(B,r.bshape),r.mb,r,α,β,ws,T,threaded)
    elseif r.mb===:copy && _colmajor(B)
        buf=_ws_tmp(ws,T,(length(B),)); _permute_copy!(buf,B,r.pb)
        _pair_core!(D,X,ta,reshape(buf,r.bshape),:N,r,α,β,ws,T,threaded); _ws_free(ws,buf)
    else
        buf=_ws_tmp(ws,T,(length(B),)); copyto!(buf,B)
        if r.mb===:copy
            buf2=_ws_tmp(ws,T,(length(B),)); _permute_copy!(buf2,buf,r.pb,size(B)); _ws_free(ws,buf)
            _pair_core!(D,X,ta,reshape(buf2,r.bshape),:N,r,α,β,ws,T,threaded); _ws_free(ws,buf2)
        else
            _pair_core!(D,X,ta,reshape(buf,r.bshape),r.mb,r,α,β,ws,T,threaded); _ws_free(ws,buf)
        end
    end
    _ws_free(ws,bufA)
    D
end
function _pair_core!(D,X,ta::Symbol,Y,tb::Symbol,r::_PairRecipe,α,β,ws,::Type{T},threaded::Bool) where T
    r.swap ? _pair_store!(D,Y,X,_flip(tb),_flip(ta),r,α,β,ws,T,threaded) :
             _pair_store!(D,X,Y,ta,tb,r,α,β,ws,T,threaded)
end
function _pair_gemm!(D3,P,Q,tx,ty,α,β,threaded::Bool;kwargs...)
    batches=size(D3,3)
    if batches==1
        T=eltype(D3)
        cfg=GemmConfig(T;parallel_threshold=_TENSOR_GEMM_PARALLEL_THRESHOLD)
        gemm!(view(D3,:,:,1),view(P,:,:,1),view(Q,:,:,1),α,β;transA=tx,transB=ty,threaded,config=cfg,kwargs...)
    else
        T=eltype(D3)
        m,n=size(D3,1),size(D3,2)
        k=tx===:N ? size(P,2) : size(P,1)
        w=_tensor_batched_workers(T,batches,m,n,k,threaded)
        if w>1
            _foreach_range(batches,w) do range,_
                for b in range
                    gemm!(view(D3,:,:,b),view(P,:,:,b),view(Q,:,:,b),α,β;transA=tx,transB=ty,threaded=false,kwargs...)
                end
            end
        else
            for b in 1:batches
                gemm!(view(D3,:,:,b),view(P,:,:,b),view(Q,:,:,b),α,β;transA=tx,transB=ty,threaded=false,kwargs...)
            end
        end
    end
    D3
end
# D (result labels r.out, the requested order) <- α·(P·Q) + β·D.
function _pair_store!(D,P,Q,tx::Symbol,ty::Symbol,r::_PairRecipe,α,β,ws,::Type{T},threaded::Bool) where T
    if isempty(r.operm) && _colmajor(D)
        _pair_gemm!(reshape(D,r.cshape),P,Q,tx,ty,α,β,threaded)   # straight into the output
    else
        tmp=_ws_tmp(ws,T,r.cshape); _pair_gemm!(tmp,P,Q,tx,ty,one(T),zero(T),threaded)
        # natural -> requested order, read from the flat GEMM result with the
        # recipe's natural dimensions (no reshape of runtime rank).
        if isone(α) && iszero(β) && _colmajor(D)
            isempty(r.operm) ? copyto!(D,tmp) : _permute_copy!(D,tmp,r.operm,r.natdims)
        else
            tmp2=_ws_tmp(ws,T,size(D)); isempty(r.operm) ? copyto!(tmp2,tmp) : _permute_copy!(tmp2,tmp,r.operm,r.natdims)
            iszero(β) ? (D .= α .* tmp2) : (D .= α .* tmp2 .+ β .* D)
            _ws_free(ws,tmp2)
        end
        _ws_free(ws,tmp)
    end
    D
end
function _einsum_run!(C,plan::EinsumPlan,arrays,α,β,ws,threaded::Bool)
    T=eltype(C); threaded=threaded && generic_threadsafe(T); rs=plan.recipes::Vector{_PairRecipe}
    if length(rs)==1   # type-stable two-operand case
        r=rs[1]; return _pair_exec!(C,arrays[r.i],arrays[r.j],r,α,β,ws,T,threaded)
    end
    data=Any[arrays...]
    for (step,r) in enumerate(rs)
        A=data[r.i]; B=data[r.j]
        if step==length(rs)
            res=_pair_exec!(C,A,B,r,α,β,ws,T,threaded)
        else
            res=_ws_tmp(ws,T,Tuple(r.natdims)); _pair_exec!(res,A,B,r,one(T),zero(T),ws,T,threaded)
        end
        _ws_free(ws,A); _ws_free(ws,B)   # intermediates only: operands never alias the pool
        deleteat!(data,r.j); deleteat!(data,r.i); push!(data,res)
    end
    C
end
# The recipe path covers dense operands of a plan that has recipes, called
# with execution options only (threaded, GEMM options): output kinds and
# sparse operands keep the general path.
_einsum_fast(plan::EinsumPlan,arrays)=plan.recipes!==nothing && !any(_is_sparse_operand,arrays)

function _einsum_compute(plan::EinsumPlan,arrays;threaded=true,allocator=nothing,release=nothing,reuse_result=false,output=:auto,kwargs...)
    threaded=threaded && generic_threadsafe(eltype(first(arrays)))
    length(arrays)==length(plan.shapes) || throw(DimensionMismatch("einsum operand count differs"))
    T=eltype(first(arrays)); T<:GenericFloat || throw(ArgumentError("unsupported einsum element type"))
    plan.memory_limit!==nothing && T!==plan.scalar_type && throw(ArgumentError("memory-bounded einsum plan scalar type changed"))
    for (A,shape) in zip(arrays,plan.shapes)
        Base.require_one_based_indexing(A)
        size(A)==shape || throw(DimensionMismatch("einsum plan shape changed"))
        eltype(A)===T || throw(ArgumentError("einsum element types must agree"))
    end
    # Sparse operands take the sparse paths (sparse_tensors.jl); dense-only
    # contractions run the dense code below unchanged.
    any(_is_sparse_operand,arrays) && return _einsum_compute_sparse(plan,arrays,T;threaded,allocator,release,reuse_result,output,kwargs...)
    output in (:auto,:dense,:sparse) || throw(ArgumentError("output must be :auto, :dense or :sparse"))
    output===:sparse && return SparseTensor(_einsum_compute(plan,arrays;threaded,allocator,release,reuse_result,kwargs...))
    final_allocator=reuse_result ? allocator : nothing
    plan.direct && length(arrays)>1 && return _einsum_direct(arrays,plan.labels,plan.output,plan.dimensions,T;threaded,allocator=final_allocator)
    data=Any[arrays...]; ls=copy(plan.labels)
    final_order=allunique(plan.output)
    for (step,(i,j,out)) in enumerate(plan.steps)
        last=step==length(plan.steps)
        pair_allocator=last ? final_allocator : allocator
        # Last step: straight into the requested output order when it has the
        # same labels (previously the planner's order, then a scalar unary
        # permutation). Intermediate steps: natural GEMM order, which the next
        # step reads without a copy.
        if last && final_order && length(out)==length(plan.output) && issetequal(out,plan.output)
            out=plan.output
            C=_einsum_pair(data[i],data[j],ls[i],ls[j],out,plan.dimensions;threaded,allocator=pair_allocator,copies=plan.memory_limit===nothing,kwargs...)
        elseif !last && allunique(out)
            C,out=_einsum_pair(data[i],data[j],ls[i],ls[j],out,plan.dimensions;threaded,allocator=pair_allocator,copies=plan.memory_limit===nothing,free_output=true,kwargs...)
        else
            C=_einsum_pair(data[i],data[j],ls[i],ls[j],out,plan.dimensions;threaded,allocator=pair_allocator,copies=plan.memory_limit===nothing,kwargs...)
        end
        if release!==nothing
            release(data[i]); release(data[j])
        end
        deleteat!(data,j); deleteat!(data,i); push!(data,C)
        deleteat!(ls,j); deleteat!(ls,i); push!(ls,out)
    end
    if ls[1]==plan.output && allunique(ls[1])
        length(arrays)>1 && return data[1]
        final_allocator===nothing && return copy(data[1])
        result=final_allocator(size(data[1])); copyto!(result,data[1]); return result
    end
    _einsum_direct((data[1],),(ls[1],),plan.output,plan.dimensions,T;threaded,allocator=final_allocator)
end

"""
    einsum(plan::EinsumPlan, arrays...; workspace=nothing, kwargs...)

Einstein summation on Julia arrays. Complex values are multiplied without
implicit conjugation. Scalar expressions return a zero-dimensional Array.
Operands may be sparse (SparseTensor, SparseMatrixCSC, SparseVector, OxiBLAS
sparse formats, and their transposes/adjoints): they are contracted without
densification. `output=:auto` returns a SparseTensor when every operand is
sparse and a dense Array otherwise; `:dense` / `:sparse` force the kind.
"""
function einsum(plan::EinsumPlan,arrays::AbstractArray...;workspace=nothing,kwargs...)
    # Dense recipe path: the result is allocated here, then filled in place
    # (einsum! semantics); the workspace only removes the temporaries.
    if all(k->k===:threaded,keys(kwargs)) && _einsum_fast(plan,arrays) && length(arrays)==length(plan.shapes) && all(A->eltype(A)===eltype(first(arrays)),arrays)
        T=eltype(first(arrays))
        if T<:GenericFloat && all(((A,s),)->size(A)==s,zip(arrays,plan.shapes))
            C=Array{T}(undef,Tuple(plan.dimensions[l] for l in plan.output))
            return einsum!(C,plan,arrays...;workspace,kwargs...)
        end
    end
    workspace===nothing && return _einsum_compute(plan,arrays;kwargs...)
    _with_einsum_workspace(workspace,arrays) do allocator,release
        _einsum_compute(plan,arrays;allocator,release,kwargs...)
    end
end
einsum(spec::AbstractString,arrays::AbstractArray...;optimize=:greedy,memory_limit=nothing,beam_width=32,search_limit=100000,cost_model=:work,layout_weight=1,kwargs...)=einsum(plan_einsum(spec,arrays...;optimize,memory_limit,beam_width,search_limit,cost_model,layout_weight),arrays...;kwargs...)

function _einsum_finish!(C,result,α,β)
    @inbounds for i in eachindex(C,result)
        C[i]=iszero(β) ? α*result[i] : α*result[i]+β*C[i]
    end
    C
end
# Shape checks without temporaries (plan.shapes holds abstract Tuples and
# plan.dimensions is a Dict: no Tuple is built per call).
function _einsum_check_call(C,plan::EinsumPlan,arrays,::Type{T}) where T
    out=plan.output
    ndims(C)==length(out) || throw(DimensionMismatch("einsum output shape differs"))
    for d in 1:length(out)
        size(C,d)==plan.dimensions[out[d]] || throw(DimensionMismatch("einsum output shape differs"))
    end
    length(arrays)==length(plan.shapes) || throw(DimensionMismatch("einsum operand count differs"))
    for k in eachindex(arrays)
        A=arrays[k]; shape=plan.shapes[k]
        Base.require_one_based_indexing(A)
        (ndims(A)==length(shape) && eltype(A)===T) || throw(DimensionMismatch("einsum operand shape/type differs"))
        for d in 1:ndims(A)
            size(A,d)==shape[d] || throw(DimensionMismatch("einsum operand shape/type differs"))
        end
    end
    nothing
end
_einsum_aliases(C,arrays)=any(A->Base.mightalias(C,A),arrays)
# Workspace execution without closures: the lease rules of
# _with_einsum_workspace (no reentry, no operand or output aliasing the pool,
# every lease returned at the end).
function _einsum_run_ws!(C,plan,arrays,α,β,ws::EinsumWorkspace{T},threaded::Bool) where T
    eltype(C)===T || throw(ArgumentError("einsum workspace scalar type differs"))
    lock(ws.guard)
    try
        ws.active && throw(ArgumentError("recursive einsum workspace use is unsupported"))
        for buffer in ws.buffers
            for A in arrays
                Base.mightalias(A,buffer) && throw(ArgumentError("operand aliases einsum workspace storage"))
            end
            Base.mightalias(C,buffer) && throw(ArgumentError("output aliases einsum workspace storage"))
        end
        ws.active=true
        try
            _einsum_run!(C,plan,arrays,α,β,ws,threaded)
        finally
            fill!(ws.leased,false); ws.active=false
        end
    finally
        unlock(ws.guard)
    end
    C
end
function einsum!(C::AbstractArray{T},plan::EinsumPlan,arrays::AbstractArray...;α=one(T),β=zero(T),workspace=nothing,threaded::Bool=true,kwargs...) where {T<:GenericFloat}
    Base.require_one_based_indexing(C)
    C isa Array || _check_nd_output(C)   # an Array is disjoint by construction
    _einsum_check_call(C,plan,arrays,T)
    plan.memory_limit!==nothing && T!==plan.scalar_type && throw(ArgumentError("memory-bounded einsum plan scalar type changed"))
    a=convert(T,α); b=convert(T,β)
    # In place, no result scratch: the last GEMM writes into C (with α, β)
    # when C's layout allows it. An output that aliases an operand keeps the
    # general path, which computes everything before writing.
    if isempty(kwargs) && _einsum_fast(plan,arrays) && !_einsum_aliases(C,arrays)
        iszero(a) && return _scale!(C,b)
        workspace===nothing && return _einsum_run!(C,plan,arrays,a,b,nothing,threaded)
        return _einsum_run_ws!(C,plan,arrays,a,b,workspace,threaded)
    end
    if workspace===nothing
        iszero(a) && return _scale!(C,b)
        return _einsum_finish!(C,einsum(plan,arrays...;output=:dense,threaded,kwargs...),a,b) # before aliased writes
    end
    _with_einsum_workspace(workspace,arrays,C) do allocator,release
        iszero(a) && return _scale!(C,b)
        _einsum_finish!(C,_einsum_compute(plan,arrays;allocator,release,reuse_result=true,output=:dense,threaded,kwargs...),a,b)
    end
end
einsum!(C,spec::AbstractString,arrays::AbstractArray...;optimize=:greedy,memory_limit=nothing,beam_width=32,search_limit=100000,cost_model=:work,layout_weight=1,kwargs...)=einsum!(C,plan_einsum(spec,arrays...;optimize,memory_limit,beam_width,search_limit,cost_model,layout_weight),arrays...;kwargs...)

function _contract_axes(A,B,axesA,axesB)
    Base.require_one_based_indexing(A,B)
    ca=axesA isa Integer ? Int[axesA] : collect(Int,axesA)
    cb=axesB isa Integer ? Int[axesB] : collect(Int,axesB)
    length(ca)==length(cb) && allunique(ca) && allunique(cb) || throw(ArgumentError("invalid contraction axes"))
    all(d->1<=d<=ndims(A),ca) && all(d->1<=d<=ndims(B),cb) || throw(ArgumentError("contraction axis out of range"))
    all(size(A,a)==size(B,b) for (a,b) in zip(ca,cb)) || throw(DimensionMismatch("contracted dimensions differ"))
    fa=setdiff(collect(1:ndims(A)),ca); fb=setdiff(collect(1:ndims(B)),cb)
    ca,cb,fa,fb
end
function contract(A::AbstractArray{T},B::AbstractArray{T},axesA,axesB;conjugateA=false,conjugateB=false,output=:auto,kwargs...) where {T<:GenericFloat}
    ca,cb,fa,fb=_contract_axes(A,B,axesA,axesB)
    (_is_sparse_operand(A) || _is_sparse_operand(B) || output===:sparse) && return _contract_sparse(A,B,ca,cb,fa,fb,conjugateA,conjugateB,output;kwargs...)
    m=prod(size(A,d) for d in fa;init=1); k=prod(size(A,d) for d in ca;init=1); n=prod(size(B,d) for d in fb;init=1)
    aa=reshape(PermutedDimsArray(A,Tuple(vcat(fa,ca))),m,k)
    bb=reshape(PermutedDimsArray(B,Tuple(vcat(cb,fb))),k,n)
    conjugateA && (aa=conj.(aa)); conjugateB && (bb=conj.(bb))
    C=Matrix{T}(undef,m,n); gemm!(C,aa,bb;kwargs...)
    reshape(C,Tuple(vcat([size(A,d) for d in fa],[size(B,d) for d in fb])))
end
function contract!(C::AbstractArray{T},A::AbstractArray{T},B::AbstractArray{T},axesA,axesB;α=one(T),β=zero(T),kwargs...) where {T<:GenericFloat}
    Base.require_one_based_indexing(C); _check_nd_output(C)
    ca,cb,fa,fb=_contract_axes(A,B,axesA,axesB)
    size(C)==Tuple(vcat([size(A,d) for d in fa],[size(B,d) for d in fb])) || throw(DimensionMismatch("contraction output shape differs"))
    α,β=convert(T,α),convert(T,β)
    (iszero(α) || any(d->size(A,d)==0,ca)) && return _scale!(C,β)
    # Computing the result first also protects arbitrary cross-view aliases.
    R=contract(A,B,axesA,axesB;output=:dense,kwargs...)
    size(C)==size(R) || throw(DimensionMismatch("contraction output shape differs"))
    @inbounds for i in eachindex(C,R)
        C[i]=iszero(β) ? α*R[i] : α*R[i]+β*C[i]
    end
    C
end
outer_product(A::AbstractArray,B::AbstractArray;kwargs...)=contract(A,B,(),();kwargs...)

"""
    mode_product(A, U, mode; kwargs...)

Multiply mode `mode` by U (new dimension × old dimension).
"""
function mode_product(A::AbstractArray,U::AbstractMatrix,mode::Integer;kwargs...)
    1<=mode<=ndims(A) || throw(ArgumentError("mode out of range"))
    C=contract(U,A,(2,),(mode,);kwargs...)
    # C order is (new mode, all other original modes).
    perm=collect(2:ndims(A)); insert!(perm,mode,1)
    permutedims(C,Tuple(perm))
end

struct TuckerFactor{T,N}
    core::Array{T,N}
    factors::Vector{Matrix{T}}
end
"""
    hosvd(A; ranks=nothing, kwargs...) -> TuckerFactor

Truncated higher-order SVD, using the native one-sided Jacobi SVD.
"""
function hosvd(A::AbstractArray{T,N};ranks=nothing,kwargs...) where {T<:GenericFloat,N}
    ranks===nothing && (ranks=ntuple(d->min(size(A,d),prod((size(A,k) for k in 1:N if k!=d);init=1)),N))
    length(ranks)==N || throw(DimensionMismatch("one Tucker rank per mode is required"))
    Base.require_one_based_indexing(A)
    factors=Matrix{T}[]
    for d in 1:N
        0<=ranks[d]<=size(A,d) || throw(ArgumentError("invalid Tucker rank"))
        perm=vcat(d,setdiff(collect(1:N),[d]))
        M=reshape(PermutedDimsArray(A,Tuple(perm)),size(A,d),:)
        S=svd_factor(M;kwargs...)
        ranks[d]<=size(S.U,2) || throw(ArgumentError("Tucker rank exceeds unfolding economy rank"))
        push!(factors,copy(S.U[:,1:ranks[d]]))
    end
    core=Array(A)
    for d in 1:N
        core=mode_product(core,adjoint(factors[d]),d)
    end
    TuckerFactor(core,factors)
end
function reconstruct(F::TuckerFactor)
    A=copy(F.core)
    for d in eachindex(F.factors)
        A=mode_product(A,F.factors[d],d)
    end
    A
end

struct TensorTrainFactor{T}
    cores::Vector{Array{T,3}}
    shape::Tuple
end
"""
    tensor_train(A; maxrank=typemax(Int), rtol=0, kwargs...) -> TensorTrainFactor

Sequential TT-SVD. `maxrank` caps each bond; `rtol` bounds the discarded
Frobenius norm per split by rtol*norm(A)/sqrt(N-1).
"""
function tensor_train(A::AbstractArray{T,N};maxrank=typemax(Int),rtol=zero(real(T)),kwargs...) where {T<:GenericFloat,N}
    N>=1 && maxrank>=1 && rtol>=0 || throw(ArgumentError("invalid tensor-train options"))
    Base.require_one_based_indexing(A)
    cores=Array{T,3}[]; rest=Array(A); previous=1
    threshold=N>1 ? (rtol*nrm2(A))^2/(N-1) : zero(real(T))
    for d in 1:N-1
        M=reshape(rest,previous*size(A,d),:); S=svd_factor(M;kwargs...)
        rank=min(maxrank,length(S.S)); discarded=sum(abs2,view(S.S,rank+1:length(S.S)))
        while rank>1 && discarded+abs2(S.S[rank])<=threshold
            discarded+=abs2(S.S[rank]); rank-=1
        end
        push!(cores,reshape(S.U[:,1:rank],previous,size(A,d),rank))
        rest=S.Vt[1:rank,:]
        for j in axes(rest,2), i in axes(rest,1)
            rest[i,j]*=S.S[i]
        end
        previous=rank
    end
    push!(cores,reshape(rest,previous,size(A,N),1))
    TensorTrainFactor(cores,size(A))
end
function reconstruct(F::TensorTrainFactor)
    A=copy(first(F.cores))
    for d in 2:length(F.cores)
        A=contract(A,F.cores[d],(ndims(A),),(1,))
    end
    reshape(A,F.shape)
end

struct CPFactor{T,R}
    weights::Vector{R}
    factors::Vector{Matrix{T}}
    iterations::Int
    relative_error::R
    converged::Bool
    reason::Symbol
end
function reconstruct(F::CPFactor{T}) where T
    shape=Tuple(size(U,1) for U in F.factors); A=zeros(T,shape)
    for index in CartesianIndices(shape), r in eachindex(F.weights)
        v=T(F.weights[r])
        for d in eachindex(F.factors)
            v*=F.factors[d][index[d],r]
        end
        A[index]+=v
    end
    A
end

"""
    cp_als(A, rank; initial=nothing, maxiter=100, rtol=..., kwargs...) -> CPFactor

CANDECOMP/PARAFAC by alternating least squares with native MTTKRP/GEMM
and LU solves. Real and complex arrays are supported. `initial` can provide
factors; otherwise deterministic, nonrandom factors are used. Convergence and
stagnation are reported separately from the attained approximation error.
"""
function cp_als(A::AbstractArray{T,N},rank::Integer;initial=nothing,maxiter=100,
                rtol=sqrt(eps(real(T))),regularization=eps(real(T)),threaded=true) where {T<:GenericFloat,N}
    threaded=threaded && generic_threadsafe(T)
    Base.require_one_based_indexing(A); R=real(T)
    N>=2 && rank>0 && all(>(0),size(A)) && maxiter>=0 && min(rtol,regularization)>=0 || throw(ArgumentError("invalid CP options"))
    factors=if initial===nothing
        [T[sin(R(i+7r+13d))+cos(R(3i+r+5d)) for i in 1:size(A,d), r in 1:rank] for d in 1:N]
    else
        length(initial)==N || throw(DimensionMismatch("one CP factor per mode is required"))
        [Matrix{T}(U) for U in initial]
    end
    all(size(factors[d])==(size(A,d),rank) for d in 1:N) || throw(DimensionMismatch("CP factor shape differs"))
    norm=nrm2(A); previous=R(Inf); error=R(Inf); iterations=0; reason=:maxiter
    G=Matrix{T}(undef,rank,rank); gram=similar(G)   # rank×rank, reused by every mode and sweep
    for iteration in 1:maxiter
        for d in 1:N
            others=setdiff(collect(1:N),[d]); shape=Tuple(size(A,k) for k in others)
            indices=CartesianIndices(shape); K=Matrix{T}(undef,length(indices),rank)
            for r in 1:rank, (linear,index) in enumerate(indices)
                v=one(T)
                for (j,k) in enumerate(others)
                    v*=conj(factors[k][index[j],r])
                end
                K[linear,r]=v
            end
            fill!(G,one(T))
            for k in others
                # transpose(F)*conj(F) = conj(F'F): one product, no conjugated copy of F.
                gemm!(gram,adjoint(factors[k]),factors[k];threaded)
                for j in 1:rank, i in 1:rank
                    G[i,j]*=conj(gram[i,j])
                end
            end
            scale=maximum(abs,G;init=one(R))
            for r in 1:rank
                G[r,r]+=regularization*scale
            end
            unfolded=reshape(PermutedDimsArray(A,Tuple(vcat(d,others))),size(A,d),:)
            rhs=Matrix{T}(undef,size(A,d),rank); gemm!(rhs,unfolded,K;threaded)
            factors[d]=Matrix(transpose(solve(lu_factor(transpose(G);threaded),Matrix(transpose(rhs)))))
            # Move norms to the following mode to preserve the represented tensor.
            following=mod1(d+1,N)
            for r in 1:rank
                s=nrm2(view(factors[d],:,r)); iszero(s) && continue
                for i in axes(factors[d],1)
                    factors[d][i,r]/=s
                end
                for i in axes(factors[following],1)
                    factors[following][i,r]*=s
                end
            end
        end
        trial=CPFactor(ones(R,rank),factors,iteration,error,false,:iterating)
        approximation=reconstruct(trial)
        for i in eachindex(approximation,A)
            approximation[i]-=A[i]
        end
        raw=nrm2(approximation); error=iszero(norm) ? raw : raw/norm; iterations=iteration
        if error<=rtol
            reason=:converged; break
        elseif isfinite(previous) && abs(previous-error)<=rtol*max(one(R),previous)
            reason=:stagnation; break
        end
        previous=error
    end
    weights=ones(R,rank)
    for U in factors, r in 1:rank
        s=nrm2(view(U,:,r)); weights[r]*=s
        iszero(s) && continue
        for i in axes(U,1)
            U[i,r]/=s
        end
    end
    CPFactor(weights,factors,iterations,error,reason===:converged,reason)
end

"""
    batched_matmul(A, B; transA=:N, transB=:N, kwargs...)

N-D batched matrix multiplication: dimensions 1/2 are matrix dimensions;
remaining dimensions broadcast independently (including singleton batch axes).
"""
function batched_matmul(A::AbstractArray{T},B::AbstractArray{T};transA=:N,transB=:N,kwargs...) where {T<:GenericFloat}
    shape=_nd_batch_shape(A,B,transA,transB)
    C=Array{T}(undef,shape)
    batched_matmul!(C,A,B;transA,transB,kwargs...)
end
function _nd_batch_shape(A,B,transA,transB)
    ndims(A)>=2 && ndims(B)>=2 || throw(ArgumentError("batched operands need at least two dimensions"))
    an=_trans_kind(transA)==0; bn=_trans_kind(transB)==0
    m,k=an ? (size(A,1),size(A,2)) : (size(A,2),size(A,1))
    kk,n=bn ? (size(B,1),size(B,2)) : (size(B,2),size(B,1))
    k==kk || throw(DimensionMismatch("batched contracted dimensions differ"))
    shape=Int[m,n]
    for d in 3:max(ndims(A),ndims(B))
        a,b=size(A,d),size(B,d)
        (a==b || a==1 || b==1) || throw(DimensionMismatch("batch dimensions do not broadcast"))
        push!(shape,a==1 ? b : a)
    end
    Tuple(shape)
end
function batched_matmul!(C::AbstractArray{T},A::AbstractArray{T},B::AbstractArray{T},α=one(T),β=zero(T);transA=:N,transB=:N,kwargs...) where {T<:GenericFloat}
    Base.require_one_based_indexing(C,A,B); _check_nd_output(C)
    size(C)==_nd_batch_shape(A,B,transA,transB) || throw(DimensionMismatch("batched output shape differs"))
    AA=Base.mightalias(C,A) ? copy(A) : A; BB=Base.mightalias(C,B) ? copy(B) : B
    indices=CartesianIndices(size(C)[3:end])
    cs=[view(C,:,:,Tuple(i)...) for i in indices]
    aa=[_op(view(AA,:,:,ntuple(d->size(AA,d+2)==1 ? 1 : i[d],ndims(AA)-2)...),transA) for i in indices]
    bb=[_op(view(BB,:,:,ntuple(d->size(BB,d+2)==1 ? 1 : i[d],ndims(BB)-2)...),transB) for i in indices]
    _batched_execute!(vec(cs),vec(aa),vec(bb),α,β;kwargs...)
    C
end
