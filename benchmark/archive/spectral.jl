using OxiBLAS, BenchmarkTools, Random, InteractiveUtils
import LinearAlgebra as LA
LA.BLAS.set_num_threads(1)
Random.seed!(20261103)
const out=get(ENV,"OXIBLAS_REPORT_DIR",joinpath(@__DIR__,"..","..","reports"))
const backend=get(ENV,"OXIBLAS_REFERENCE","openblas")
if backend=="accelerate"
    @eval using AppleAccelerate
    AppleAccelerate.set_num_threads(1)
end
open(joinpath(out,"spectral-$(backend)-environment.txt"),"w") do io
    versioninfo(io)
    println(io,"BLAS/LAPACK reference: ",LA.BLAS.get_config())
    println(io,"Reference threads: ",LA.BLAS.get_num_threads())
    println(io,"BenchmarkTools ",pkgversion(BenchmarkTools))
    println(io,"Native spectral routines are sequential; compact WY uses native GEMM with one worker.")
    println(io,"Trials: median/minimum, evals=1, max 7 samples, 0.12 seconds; warmup and correctness outside timing.")
    println(io,"No GFLOP estimate: operations have data-dependent iteration counts.")
end

function measure(io,op,T,m,n,method,f,verify)
    first=@timed f()
    residual,orthogonality,iterations=verify(first.value)
    R=real(T)
    limit=200eps(R)*max(1,m,n)
    residual<=limit && orthogonality<=limit || error("spectral benchmark correctness failed: $((op,T,m,n,method,residual,orthogonality,limit))")
    trial=run(@benchmarkable($f(),evals=1);seconds=0.12,samples=7)
    med=BenchmarkTools.median(trial).time/1e9; small=minimum(trial).time/1e9
    compile=hasproperty(first,:compile_time) ? first.compile_time : NaN
    println(io,join((op,T,m,n,method,med,small,trial.memory,trial.allocs,length(trial),first.time,compile,residual,orthogonality,iterations),','));flush(io)
    println((op,T,m,n,method,round(med;digits=6)));flush(stdout)
end

open(joinpath(out,"spectral-$(backend).csv"),"w") do io
    println(io,"operation,type,m,n,algorithm,median_s,minimum_s,bytes,allocations,samples,first_call_s,compile_s,relative_residual,orthogonality,iterations")
    for T in (Float32,Float64,ComplexF32,ComplexF64), n in (16,64,128,256)
        A=randn(T,n,n); scale=max(LA.norm(A),eps(real(T)))
        verify_h(F)=(LA.norm(A-F.Q*F.H*F.Q')/scale,LA.norm(F.Q'*F.Q-LA.I),0)
        measure(io,:hessenberg,T,n,n,:lapack,()->LA.hessenberg(A),verify_h)
        verify_s(F)=(LA.norm(A-F.Z*F.T*F.Z')/scale,LA.norm(F.Z'*F.Z-LA.I),hasproperty(F,:iterations) ? F.iterations : -1)
        measure(io,:schur,T,n,n,:lapack,()->LA.schur(A),verify_s)
        H=Matrix(LA.Hermitian(A)); hnorm=max(LA.norm(H),eps(real(T)))
        verify_e(F)=(LA.norm(H*F.vectors-F.vectors*LA.Diagonal(F.values))/hnorm,LA.norm(F.vectors'*F.vectors-LA.I),hasproperty(F,:iterations) ? F.iterations : -1)
        measure(io,:hermitian_eigen,T,n,n,:lapack,()->LA.eigen(LA.Hermitian(H)),verify_e)
        if backend=="openblas"
            measure(io,:hessenberg,T,n,n,:native,()->hessenberg_factor(A),verify_h)
            measure(io,:schur,T,n,n,:native,()->schur_factor(A),verify_s)
            measure(io,:hermitian_eigen,T,n,n,:native,()->eigen_hermitian(H),verify_e)
            F=hessenberg_factor(A); expected=F.Q
            verify_q(Q)=(LA.norm(Q-expected)/max(1,LA.norm(expected)),LA.norm(Q'*Q-LA.I),0)
            measure(io,:hessenberg_q,T,n,n,:unblocked,()->hessenberg_q(F;algorithm=:unblocked),verify_q)
            measure(io,:hessenberg_q,T,n,n,:blocked,()->hessenberg_q(F;algorithm=:blocked),verify_q)
        end
        # Economy SVD, with a rectangular workload in addition to square.
        for m in (n,2n)
            M=randn(T,m,n); normM=max(LA.norm(M),eps(real(T)))
            verify_v(F)=(LA.norm(M-F.U*LA.Diagonal(F.S)*F.Vt)/normM,
                         max(LA.norm(F.U'*F.U-LA.I),LA.norm(F.Vt*F.Vt'-LA.I)),
                         hasproperty(F,:sweeps) ? F.sweeps : -1)
            measure(io,:svd,T,m,n,:lapack,()->LA.svd(M;full=false),verify_v)
            backend=="openblas" && measure(io,:svd,T,m,n,:jacobi,()->svd_factor(M),verify_v)
        end
    end
end
