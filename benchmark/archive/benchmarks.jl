using OxiBLAS, BenchmarkTools, Random, InteractiveUtils
import LinearAlgebra as LA

Random.seed!(20261102)
LA.BLAS.set_num_threads(1)
const OUT=get(ENV,"OXIBLAS_REPORT_DIR",joinpath(@__DIR__,"..","..","reports"))
mkpath(OUT)
const backend=get(ENV,"OXIBLAS_REFERENCE","openblas")
if backend=="accelerate"
    @eval using AppleAccelerate
    AppleAccelerate.set_num_threads(1)
elseif backend!="openblas"
    error("unknown reference backend")
end
const large="--large" in ARGS
const smoke="--smoke" in ARGS
const report=joinpath(OUT,"gemm-$(backend)-$(Threads.nthreads())t$(large ? "-large" : smoke ? "-smoke" : "").csv")
open(replace(report,".csv"=>"-environment.txt"),"w") do io
    versioninfo(io)
    println(io,"BLAS: ",LA.BLAS.get_config())
    println(io,"BLAS threads: ",LA.BLAS.get_num_threads())
    println(io,"BenchmarkTools: ",pkgversion(BenchmarkTools))
    println(io,"Timing: median/minimum, evals=1, 0.15s budget, at most 15 samples.")
    println(io,"Compilation: Julia @timed compile_time, separately from warmed trials.")
    println(io,"Workload: alpha=0.7, beta=0; preallocated output and packing workspace.")
    println(io,"FLOPs: real=2mnk, complex=8mnk (conventional leading-order counts).")
    println(io,"Memory traffic/hardware counters: not measured.")
end

function benchcase(io,T,m,n,k,method,threaded)
    if backend=="openblas"
        LA.BLAS.set_num_threads(method===:reference_blas && threaded ? Threads.nthreads() : 1)
    end
    A=randn(T,m,k); B=randn(T,k,n); C=zeros(T,m,n)
    α=T(0.7); β=zero(T)
    config=GemmConfig(T)
    w=GemmWorkspace(T,m,n,k;workers=threaded ? Threads.nthreads() : 1,config)
    firstcall=if method===:reference_blas
        @timed LA.mul!(C,A,B,α,β)
    else
        @timed gemm!(C,A,B,α,β;algorithm=method,threaded,workspace=w,config)
    end
    expected=α*(A*B)
    R=real(T)
    relative_error=LA.norm(C-expected)/max(LA.norm(expected),eps(R))
    relative_error<=100*eps(R)*max(1,k) || error("benchmark verification failed")
    bench=if method===:reference_blas
        @benchmarkable LA.mul!($C,$A,$B,$α,$β) evals=1
    else
        @benchmarkable gemm!($C,$A,$B,$α,$β;algorithm=$method,threaded=$threaded,workspace=$w,config=$config) evals=1
    end
    trial=run(bench;seconds=0.15,samples=15)
    median_ns=BenchmarkTools.median(trial).time
    minimum_ns=minimum(trial).time
    flops=(T<:Real ? 2.0 : 8.0)*m*n*k
    compile_time=hasproperty(firstcall,:compile_time) ? firstcall.compile_time : NaN
    active_workers=if method===:reference_blas
        backend=="accelerate" ? AppleAccelerate.get_num_threads() : LA.BLAS.get_num_threads()
    elseif threaded && Float64(m)*n*k>=config.parallel_threshold
        mr,nr=kernel_info(T).mr,kernel_info(T).nr
        nt=Threads.nthreads()
        min(nt,cld(n,nr)>=nt ? cld(n,nr) : cld(m,mr))
    else
        1
    end
    fields=(string(T),m,n,k,method,threaded ? Threads.nthreads() : 1,active_workers,
            median_ns/1e9,minimum_ns/1e9,flops/median_ns,trial.memory,trial.allocs,
            length(trial),firstcall.time,compile_time,relative_error)
    println(io,join(fields,',')); flush(io)
    println((T,m,n,k,method,threaded,round(flops/median_ns;digits=2),"GFLOP/s"))
    flush(stdout)
    backend=="openblas" && LA.BLAS.set_num_threads(1)
end

open(report,"w") do io
    println(io,"type,m,n,k,algorithm,threads,active_workers,median_s,minimum_s,gflops,bytes,allocations,samples,first_call_s,compile_s,relative_error")
    for T in (Float32,Float64,ComplexF32,ComplexF64)
        sizes=smoke ? (16,64,256) : large ? (16,32,64,128,256,512,1024,2048,4096) : (16,32,64,128,256,512,1024)
        for n in sizes
            methods=backend=="accelerate" ? (:reference_blas,) : n<=128 ? (:reference,:blocked,:packed,:register,:simd,:reference_blas) : n<=512 ? (:blocked,:register,:simd,:reference_blas) : (:simd,:reference_blas)
            for method in methods
                benchcase(io,T,n,n,n,method,false)
            end
            if Threads.nthreads()>1 && backend=="openblas"
                benchcase(io,T,n,n,n,:simd,true)
                benchcase(io,T,n,n,n,:reference_blas,true)
            end
        end
        if !smoke
            for (m,n,k) in ((2048,32,256),(32,2048,256),(127,193,79),(256,256,2048))
                benchcase(io,T,m,n,k,:reference_blas,false)
                if backend=="openblas"
                    benchcase(io,T,m,n,k,:simd,false)
                    Threads.nthreads()>1 && benchcase(io,T,m,n,k,:simd,true)
                    Threads.nthreads()>1 && benchcase(io,T,m,n,k,:reference_blas,true)
                end
            end
        end
    end
end
println("Saved ",report)
