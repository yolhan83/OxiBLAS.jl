# Prepared only. Must be explicitly authorized before execution.
using OxiBLAS, BenchmarkTools, Random, InteractiveUtils
import LinearAlgebra as LA
length(ARGS)==1 || error("usage: block_spectral.jl NEW_OUTPUT_DIRECTORY")
out=abspath(ARGS[1]); ispath(out) && error("refusing to overwrite existing results")
mkpath(out); Random.seed!(825); LA.BLAS.set_num_threads(1)
open(joinpath(out,"environment.txt"),"w") do io
    versioninfo(io;verbose=true)
    println(io,"OxiBLAS=",pkgversion(OxiBLAS),"; native threads=",Threads.nthreads())
    println(io,"Reference backend=",LA.BLAS.get_config(),"; reference threads=",LA.BLAS.get_num_threads())
    println(io,"Seed=825; samples<=7; seconds<=0.15/case; evals=1; samples may exceed time budget.")
    println(io,"Projection timing is separate from complete solver timing. Failed guards abort before timing.")
end
function block_measure(io,name,T,n,f,verify)
    first=@timed f(); checks=verify(first.value)
    checks.ok || error("independent guard failed: $((name,T,n,checks))")
    trial=run(@benchmarkable($f(),evals=1);seconds=0.15,samples=7)
    compile=hasproperty(first,:compile_time) ? first.compile_time : NaN
    retained=Base.summarysize(first.value)
    println(io,join((name,T,n,Threads.nthreads(),checks.residual,checks.value_error,
                    checks.dimension,checks.iterations,checks.operator_work,retained,
                    first.time,compile,BenchmarkTools.median(trial).time/1e9,
                    minimum(trial).time/1e9,trial.memory,trial.allocs,length(trial)),','))
    flush(io)
end
open(joinpath(out,"block-spectral.csv"),"w") do io
    println(io,"operation,type,n,threads,relative_residual,relative_value_error,basis_dimension,iterations,operator_work,retained_result_bytes,first_s,compile_s,median_s,min_s,bytes,allocations,samples")
    for T in (Float32,Float64,ComplexF32,ComplexF64), n in (32,96)
        R=real(T); d=R.(range(1,4;length=n)); d[end-1]=R(12); d[end]=R(16)
        A=CSRMatrix(Matrix(LA.Diagonal(T.(d)))); dense=Matrix(A); limit=500n*eps(R)
        function projection_check(F)
            relation=LA.norm(dense*F.basis-F.basis*F.projected-F.residual_basis*F.residual_coupling)/LA.norm(dense)
            orthogonality=LA.norm(F.basis'*F.basis-LA.I)
            (;ok=relation<=limit && orthogonality<=limit,residual=relation,value_error=NaN,
              dimension=size(F.basis,2),iterations=F.iterations,operator_work=F.matvecs)
        end
        block_measure(io,:block_lanczos_projection,T,n,
                      ()->block_lanczos_factor(A;blocksize=4,maxblocks=4,subspace=24),projection_check)
        function eigen_check(result)
            residual=LA.norm(dense*result.vectors-result.vectors*LA.Diagonal(result.values))/LA.norm(dense)
            value_error=LA.norm(result.values-R[16,12])/LA.norm(R[16,12])
            (;ok=result.converged && residual<=max(limit,10sqrt(eps(R))) && value_error<=max(limit,10sqrt(eps(R))),
              residual,value_error,dimension=result.basis_dimension,iterations=result.iterations,
              operator_work=result.matvecs+result.verification_matvecs)
        end
        block_measure(io,:block_lanczos_full,T,n,
                      ()->block_lanczos(A;nev=2,which=:largest,blocksize=4,maxblocks=n,subspace=n),eigen_check)
        block_measure(io,:thick_restart,T,n,
                      ()->thick_restart_lanczos(A;nev=2,which=:largest,subspace=16,keep=4,maxrestarts=30),eigen_check)
        # A non-Hermitian matrix with independent LAPACK reference eigenvalues.
        N=T.(dense)
        for k in 1:2:n-1
            N[k,k+1]=T(-0.125); N[k+1,k]=T(0.125)
        end
        reference=LA.eigvals(N); selected=sortperm(abs.(reference);rev=true)[1:2]
        expected=reference[selected]
        function arnoldi_check(result)
            residual=LA.norm(N*result.vectors-result.vectors*LA.Diagonal(result.values))/LA.norm(N)
            # Assignment independent of conjugate-pair ordering.
            error1=LA.norm(result.values-expected); error2=LA.norm(result.values-reverse(expected))
            value_error=min(error1,error2)/LA.norm(expected)
            (;ok=result.converged && residual<=max(limit,10sqrt(eps(R))) && value_error<=max(limit,10sqrt(eps(R))),
              residual,value_error,dimension=result.basis_dimension,iterations=result.iterations,
              operator_work=result.matvecs+result.verification_matvecs)
        end
        block_measure(io,:block_arnoldi_full,T,n,
                      ()->block_arnoldi(CSRMatrix(N);nev=2,blocksize=4,maxblocks=n,subspace=n),arnoldi_check)
        # Rectangular sparse singular values with an analytic independent reference.
        rectangular=zeros(T,n+7,n)
        for k in 1:n
            rectangular[k,k]=T(d[k])*(T<:Complex ? T(cis(R(k)/7)) : one(T))
        end
        sparse=CSRMatrix(rectangular)
        function svd_check(result)
            left=LA.norm(rectangular*result.Vt'-result.U*LA.Diagonal(result.S))
            right=LA.norm(rectangular'*result.U-result.Vt'*LA.Diagonal(result.S))
            residual=max(left,right)/LA.norm(rectangular); value_error=LA.norm(result.S-R[16,12])/20
            (;ok=result.converged && residual<=max(limit,10sqrt(eps(R))) && value_error<=max(limit,10sqrt(eps(R))),
              residual,value_error,dimension=result.krylov.basis_dimension,iterations=result.iterations,
              operator_work=result.operator_matvecs+result.verification_matvecs)
        end
        block_measure(io,:sparse_svd_dilation,T,n,
                      ()->sparse_svd_lanczos(sparse,2;algorithm=:dilation,subspace=20,keep=4,maxrestarts=30),svd_check)
    end
end
