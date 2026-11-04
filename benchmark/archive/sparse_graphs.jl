# Prepared only; no execution before approval of a named campaign.
using OxiBLAS, BenchmarkTools, Random, InteractiveUtils, SHA
import LinearAlgebra as LA
length(ARGS)==1 || error("usage: sparse_graphs.jl NEW_OUTPUT_DIRECTORY")
out=abspath(ARGS[1]); ispath(out) && error("refusing to overwrite graph results")
mkpath(out); Random.seed!(828); LA.BLAS.set_num_threads(1)
open(joinpath(out,"environment.txt"),"w") do io
    versioninfo(io;verbose=true)
    println(io,"OxiBLAS=",pkgversion(OxiBLAS),"; native threads=",Threads.nthreads())
    println(io,"Reference backend=",LA.BLAS.get_config(),"; reference threads=",LA.BLAS.get_num_threads())
    println(io,"Data seed=828; <=7 samples; <=0.15 seconds/case; evals=1; first calls may exceed budget.")
    println(io,"Graph algorithms are sequential; multiple thread counts check overhead/concurrent safety, not parallel speedup.")
    println(io,"Independent guards; Rust provider prepared as a separately authorized process; see rust_graph_reference/.")
end
open(joinpath(out,"sources.sha256"),"w") do io
    root=dirname(dirname(@__DIR__))
    for dir in ("src","test","benchmark"), (path,_,files) in walkdir(joinpath(root,dir)), name in sort(files)
        endswith(name,".jl") || endswith(name,".rs") || endswith(name,".toml") || continue
        full=joinpath(path,name); println(io,bytes2hex(sha256(read(full))),"  ",relpath(full,root))
    end
end
# FNV-1a over zero-based CSR metadata and little-endian Float64 bits; same
# protocol as the prepared Rust reference, solely for checking equal inputs.
function graph_csr_fingerprint(A::CSRMatrix{Float64})
    hash=UInt64(0xcbf29ce484222325)
    words=vcat(UInt64[size(A,1),size(A,2)],UInt64.(A.rowptr.-1),UInt64.(A.colind.-1),reinterpret(UInt64,A.values))
    for word in words, shift in 0:8:56
        hash=(hash ⊻ UInt64((word>>shift)&0xff))*UInt64(0x100000001b3)
    end
    string(hash;base=16,pad=16)
end
write(joinpath(out,"inputs.sha256"),"")
write(joinpath(out,"cases-fnv64.csv"),"pattern,type,n,csr_fnv64\n")
function graph_measure(io,name,T,n,f,verify)
    first=@timed f(); check=verify(first.value)
    check.ok || error("graph benchmark guard failed: $((name,T,n,check))")
    trial=run(@benchmarkable($f(),evals=1);seconds=0.15,samples=7)
    compile=hasproperty(first,:compile_time) ? first.compile_time : NaN
    println(io,join((name,T,n,Threads.nthreads(),check.metric,first.time,compile,
                    BenchmarkTools.median(trial).time/1e9,minimum(trial).time/1e9,
                    trial.memory,trial.allocs,length(trial),Base.summarysize(first.value)),','))
    flush(io)
end
open(joinpath(out,"sparse-graphs.csv"),"w") do io
    println(io,"operation,type,n,threads,quality_metric,first_s,compile_s,median_s,min_s,bytes,allocations,samples,retained_result_bytes")
    for T in (Float32,Float64,ComplexF32,ComplexF64), n in (32,96)
        assignment=zeros(T,n,n)
        for i in 1:n
            assignment[i,i]=T(10); assignment[i,mod1(i+1,n)]=T(3)
        end
        C=CSRMatrix(assignment)
        graph_measure(io,:weighted_assignment,T,n,()->weighted_bipartite_matching(C;weight=:abs),
                      result->(;ok=result.rows==collect(1:n) && result.cardinality==n && abs(result.total_weight-10n)<1e-8,metric=result.total_weight))
        star=zeros(T,n,n)
        for j in 2:n
            star[1,j]=star[j,1]=T(-1)
        end
        for i in 1:n
            star[i,i]=T(2+count(!iszero,star[i,:]))
        end
        A=CSRMatrix(star)
        function ordering_check(p)
            valid=sort(p)==collect(1:n)
            metric=valid ? ordering_metrics(A,p).factor_nnz : typemax(Int)
            (;ok=valid && metric==2n-1,metric)
        end
        graph_measure(io,:minimum_degree,T,n,()->minimum_degree_ordering(A),ordering_check)
        graph_measure(io,:level_dissection_star,T,n,()->nested_dissection(A),ordering_check)
        graph_measure(io,:multiple_minimum_degree,T,n,()->multiple_minimum_degree(A),ordering_check)
        graph_measure(io,:column_elements,T,n,()->column_minimum_degree(A),ordering_check)
        graph_measure(io,:column_elements_dense_filter,T,n,()->column_minimum_degree(A;dense_threshold=.5),ordering_check)
        graph_measure(io,:multilevel_dissection,T,n,()->multilevel_nested_dissection(A;leafsize=8,coarse_size=8),ordering_check)
        graph_measure(io,:select_ordering,T,n,
                      ()->select_sparse_ordering(A;algorithms=(:natural,:minimum_degree,:multilevel_nested_dissection)),
                      result->(;ok=result.best.factor_nnz==2n-1,metric=result.best.factor_nnz))
        path=zeros(T,n,n)
        for i in 1:n-1
            path[i,i+1]=one(T)
        end
        P=CSRMatrix(path); symmetric_path=path+transpose(path); P2=CSRMatrix(symmetric_path)
        graph_measure(io,:weighted_graph_matching,T,n,()->weighted_graph_matching(P;weight=:abs),
                      result->(;ok=result!==nothing && result.cardinality==n÷2 && result.total_weight==n÷2,metric=result===nothing ? NaN : result.total_weight))
        graph_measure(io,:multilevel_partition_path,T,n,()->multilevel_partition(P2,4;coarse_size=8,refine_passes=6),
                      r->(;ok=length(r.labels)==n && all(!isempty,r.parts) && r.cut==count(i->r.labels[i]!=r.labels[i+1],1:n-1),metric=r.cut))
        graph_measure(io,:graph_cardinality_matching,T,n,()->bipartite_graph_matching(P),
                      result->(;ok=result.cardinality==n÷2,metric=result.cardinality))
        graph_measure(io,:matrix_cardinality_matching,T,n,()->bipartite_matching(C),
                      result->(;ok=result.cardinality==n,metric=result.cardinality))
        graph_measure(io,:coloring,T,n,()->bipartite_coloring(P),
                      result->(;ok=result.is_bipartite && all(result.colors[i]!=result.colors[i+1] for i in 1:n-1),metric=length(result.left)))
        graph_measure(io,:pseudo_peripheral,T,n,()->pseudo_peripheral_vertex(P;start=n÷2),
                      result->(;ok=result in (1,n),metric=result))
        # Sparse grid-like pattern; quality is reported, never assumed optimal.
        grid=copy(path)
        for i in 1:n-6
            grid[i,i+6]=one(T)
        end
        G=CSRMatrix(grid)
        for (pattern,data) in ((:assignment,assignment),(:star,star),(:path,path),(:symmetric_path,symmetric_path),(:grid,grid))
            open(joinpath(out,"inputs.sha256"),"a") do hashes
                println(hashes,bytes2hex(sha256(reinterpret(UInt8,vec(data)))),"  ",pattern,"-",T,"-",n)
            end
            if T===Float64
                open(joinpath(out,"cases-fnv64.csv"),"a") do cases
                    println(cases,join((pattern,T,n,graph_csr_fingerprint(CSRMatrix(data))),','))
                end
            end
        end
        function partition_check(result)
            cut=sum((1 for i in 1:n for j in i+1:n if result.labels[i]!=result.labels[j] && !iszero(grid[i,j]));init=0)
            valid=sort(vcat(result.parts...))==collect(1:n) && all(!isempty,result.parts)
            loads=Float64.(length.(result.parts))
            (;ok=valid && result.cut==cut && result.loads==loads,metric=result.cut)
        end
        for refinement in (:greedy,:fm)
            graph_measure(io,Symbol("multilevel_partition_",refinement),T,n,
                          ()->multilevel_partition(G,4;coarse_size=8,refinement),partition_check)
            graph_measure(io,Symbol("multilevel_dissection_",refinement),T,n,
                          ()->multilevel_nested_dissection(G;leafsize=8,coarse_size=8,refinement),
                          p->(;ok=sort(p)==collect(1:n),metric=ordering_metrics(G,p).factor_nnz))
        end
        initial=[i<=n÷2 ? 1 : 2 for i in 1:n]
        for method in (:greedy,:fm)
            function refinement_check(result)
                cut=sum((1 for i in 1:n for j in i+1:n if result.labels[i]!=result.labels[j] && !iszero(grid[i,j]));init=0)
                original_cut=sum((1 for i in 1:n for j in i+1:n if initial[i]!=initial[j] && !iszero(grid[i,j]));init=0)
                (;ok=result.cut==cut && result.balanced && cut<=original_cut,metric=cut)
            end
            graph_measure(io,Symbol("refinement_",method),T,n,()->refine_partition(G,initial;method),refinement_check)
        end
        graph_measure(io,:level_dissection,T,n,()->nested_dissection(G;leafsize=8,maxdepth=50),
                      p->(;ok=sort(p)==collect(1:n),metric=ordering_metrics(G,p).factor_nnz))
        graph_measure(io,:components,T,n,()->connected_components(G),
                      r->(;ok=r.count==1 && length(r.components[1])==n,metric=r.count))
        graph_measure(io,:levels,T,n,()->level_sets(P,1),
                      r->(;ok=length(r)==n && all(length(v)==1 for v in r),metric=length(r)))
        graph_measure(io,:degrees,T,n,()->degree_sequence(P),
                      r->(;ok=r==vcat(1,fill(2,n-2),1),metric=sum(r)))
        graph_measure(io,:structural_symmetry,T,n,()->is_structurally_symmetric(P),
                      r->(;ok=!r,metric=Int(r)))
        graph_measure(io,:graph_bandwidth,T,n,()->graph_bandwidth_profile(P),
                      r->(;ok=r.bandwidth==1 && r.profile==n-1,metric=r.profile))
        graph_measure(io,:symbolic_pattern,T,n,()->symbolic_cholesky(P),
                      r->(;ok=r.nnz==2n-1 && sort(r.postorder)==collect(1:n),metric=r.nnz))
    end
end

open(joinpath(out,"results.sha256"),"w") do io
    for name in sort(readdir(out))
        name=="results.sha256" && continue
        println(io,bytes2hex(sha256(read(joinpath(out,name)))),"  ",name)
    end
end
