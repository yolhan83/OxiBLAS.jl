# Prepared only. Requires an explicitly authorized supervised campaign.
using OxiBLAS, BenchmarkTools, InteractiveUtils
import LinearAlgebra as LA
length(ARGS)==1 || error("usage: stochastic.jl NEW_OUTPUT_DIRECTORY")
out=abspath(ARGS[1]); ispath(out) && error("output path already exists; refusing overwrite")
mkpath(out); LA.BLAS.set_num_threads(1)
open(joinpath(out,"environment.txt"),"w") do io
    versioninfo(io;verbose=true)
    println(io,"OxiBLAS version: ",pkgversion(OxiBLAS))
    println(io,"Threads: ",Threads.nthreads(),"; BLAS reference threads: ",LA.BLAS.get_num_threads())
    println(io,"Reference backend: ",LA.BLAS.get_config())
    println(io,"seed=824; samples<=7; seconds<=0.15/case; evals=1")
    println(io,"Independent analytic Toeplitz references; no stochastic reference timing provider yet.")
end
function measure_stochastic(io,name,T,n,f,reference,extract)
    first=@timed f(); estimate,se=extract(first.value)
    # These are gross correctness guards, not confidence-interval certification.
    allowance=isfinite(se) ? 8se+0.08abs(reference) : 0.25abs(reference)
    isfinite(estimate) && abs(estimate-reference)<=allowance || error("analytic guard failed: $((name,T,n,estimate,se,reference))")
    trial=run(@benchmarkable($f(),evals=1);seconds=0.15,samples=7)
    med=BenchmarkTools.median(trial)
    compile=hasproperty(first,:compile_time) ? first.compile_time : NaN
    println(io,join((name,T,n,Threads.nthreads(),estimate,se,reference,
                    first.value.matvecs,first.time,compile,med.time/1e9,
                    minimum(trial).time/1e9,trial.memory,trial.allocs,length(trial)),','))
    flush(io)
end
open(joinpath(out,"stochastic.csv"),"w") do io
    println(io,"operation,type,n,threads,estimate,sampling_se,analytic_reference,matvecs,first_s,compile_s,median_s,min_s,bytes,allocations,samples")
    for T in (Float32,Float64,ComplexF32,ComplexF64), n in (64,512,2048)
        R=real(T); rows=Int[]; cols=Int[]; values=T[]
        for i in 1:n
            push!(rows,i); push!(cols,i); push!(values,T(4))
            if i<n
                append!(rows,(i,i+1)); append!(cols,(i+1,i)); append!(values,(T(-1),T(-1)))
            end
        end
        A=CSRMatrix(COOMatrix(n,n,rows,cols,values))
        config=StochasticConfig(num_probes=30,seed=824)
        basic=r->(real(r.estimate),r.std_error)
        for (name,f) in ((:hutchinson,()->trace_hutchinson(A;config)),
                         (:hutchpp,()->trace_hutch_plus_plus(A;config)),
                         (:xtrace_loo,()->trace_xtrace_loo(A;config=StochasticConfig(num_probes=8,seed=824))),
                         (:xtrace_optimized,()->trace_xtrace(A;config=StochasticConfig(num_probes=8,seed=824))))
            measure_stochastic(io,name,T,n,f,4n,basic)
        end
        measure_stochastic(io,:diagonal_sum,T,n,()->stochastic_diagonal(A;config),4n,
                           r->(real(sum(r.diagonal)),sum(r.std_error)))
        measure_stochastic(io,:frobenius_squared,T,n,()->stochastic_frobenius(A;config),18n-2,
                           r->(r.squared_estimate,r.squared_std_error))
        logdet=sum(log(4-2cospi(k/(n+1))) for k in 1:n)
        measure_stochastic(io,:logdet_slq,T,n,()->stochastic_logdet(A;config,lanczos_steps=24),logdet,basic)
    end
end
