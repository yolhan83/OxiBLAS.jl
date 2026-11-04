# Prepared only; execute only as part of a separately authorized campaign.
using OxiBLAS, BenchmarkTools, Random, InteractiveUtils
import LinearAlgebra as LA
length(ARGS)==1 || error("usage: tensor_plans.jl NEW_OUTPUT_DIRECTORY")
out=abspath(ARGS[1]); ispath(out) && error("refusing to overwrite an existing output path")
mkpath(out); Random.seed!(824); LA.BLAS.set_num_threads(1)
open(joinpath(out,"environment.txt"),"w") do io
    versioninfo(io;verbose=true)
    println(io,"Reference: ",LA.BLAS.get_config(),"; reference threads=",LA.BLAS.get_num_threads())
    println(io,"Native threads=",Threads.nthreads(),"; OxiBLAS=",pkgversion(OxiBLAS))
    println(io,"Seed=824; evals=1; <=7 samples; <=0.15 s/case, one sample may exceed budget.")
    println(io,"Tensor payload bounds exclude metadata, BLAS packing, retained inactive pool capacity and RSS.")
end
function tensor_measure(io,name,T,n,plan,workspace,f,reference)
    first=@timed f(); result=first.value
    error=LA.norm(result-reference)/max(LA.norm(reference),eps(real(T)))
    error<=200n*eps(real(T)) || Base.error("tensor reference failed: $((name,T,n,error))")
    trial=run(@benchmarkable($f(),evals=1);seconds=0.15,samples=7)
    retained=workspace===nothing ? 0 : sum(length,workspace.buffers;init=0)*sizeof(T)
    compile=hasproperty(first,:compile_time) ? first.compile_time : NaN
    layout=einsum_layout_report(plan)
    println(io,join((name,T,n,Threads.nthreads(),(plan isa SlicedEinsumPlan ? plan.base.direct : plan.direct),plan.peak_intermediate_bytes,
                    retained,first.time,compile,BenchmarkTools.median(trial).time/1e9,
                    minimum(trial).time/1e9,trial.memory,trial.allocs,length(trial),error,layout.cost_model,layout.estimated_work,layout.estimated_traffic),','))
    flush(io)
end
open(joinpath(out,"tensor-plans.csv"),"w") do io
    println(io,"case,type,n,threads,direct,planned_peak_payload_bytes,retained_pool_bytes,first_s,compile_s,median_s,min_s,bytes,allocations,samples,relative_error,cost_model,estimated_work,estimated_traffic")
    for T in (Float32,Float64,ComplexF32,ComplexF64), n in (8,16,32)
        A=randn(T,n,n); B=randn(T,n,n); C=randn(T,n,n)
        reference=permutedims(A*B*C); output=similar(reference)
        for optimize in (:greedy,:left,:beam,:optimal)
            plan=plan_einsum("ij,jk,kl->li",A,B,C;optimize)
            tensor_measure(io,Symbol("planned_",optimize),T,n,plan,nothing,
                           ()->einsum(plan,A,B,C),reference)
        end
        for optimize in (:greedy,:beam,:optimal)
            physical=plan_einsum("ij,jk,kl->li",A,B,C;optimize,cost_model=:layout)
            tensor_measure(io,Symbol("layout_",optimize),T,n,physical,nothing,
                           ()->einsum(physical,A,B,C),reference)
        end
        plan=plan_einsum("ij,jk,kl->li",A,B,C)
        workspace=EinsumWorkspace(T)
        tensor_measure(io,:workspace,T,n,plan,workspace,
                       ()->einsum!(output,plan,A,B,C;workspace),reference)
        bounded=plan_einsum("ij,jk,kl->li",A,B,C;memory_limit=n*n*sizeof(T))
        tensor_measure(io,:bounded_direct,T,n,bounded,nothing,
                       ()->einsum(bounded,A,B,C),reference)
        tensor_measure(io,:bounded_workspace,T,n,bounded,workspace,
                       ()->einsum!(output,bounded,A,B,C;workspace),reference)
        sliced=plan_einsum_sliced("ij,jk,kl->li",A,B,C;chunks=(i=max(1,n÷2),j=max(1,n÷2)),scratch_limit=2n*n*sizeof(T))
        tensor_measure(io,:sliced_workspace,T,n,sliced,workspace,
                       ()->einsum!(output,sliced,A,B,C;workspace),reference)
        # Include planning/call overhead as distinct cases, with correctness first.
        tensor_measure(io,:replan_each_call,T,n,plan,nothing,
                       ()->einsum("ij,jk,kl->li",A,B,C),reference)
    end
end
