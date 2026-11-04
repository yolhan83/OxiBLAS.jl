# Prepared only; requires approval of a named execution campaign.
using OxiBLAS, BenchmarkTools, Random, InteractiveUtils
import LinearAlgebra as LA
length(ARGS)==1 || error("usage: interval_spectral.jl NEW_OUTPUT_DIRECTORY")
out=abspath(ARGS[1]); ispath(out) && error("refusing to overwrite results")
mkpath(out); Random.seed!(827); LA.BLAS.set_num_threads(1)
open(joinpath(out,"environment.txt"),"w") do io
    versioninfo(io;verbose=true)
    println(io,"OxiBLAS=",pkgversion(OxiBLAS),"; native threads=",Threads.nthreads())
    println(io,"Reference backend=",LA.BLAS.get_config(),"; reference threads=",LA.BLAS.get_num_threads())
    println(io,"Data seed=827; Krylov seed=0x0b1a5; <=7 samples; <=0.15s/case; evals=1.")
    println(io,"Analytic Toeplitz spectrum and sine basis; no timing is accepted after a failed guard.")
    println(io,"Reusable filter workspace is warmed before hot timing; allocations may include metadata/packing.")
end
function interval_measure(io,name,T,n,f,verify)
    first=@timed f(); checks=verify(first.value)
    checks.ok || error("spectral interval guard failed: $((name,T,n,checks))")
    trial=run(@benchmarkable($f(),evals=1);seconds=0.15,samples=7)
    compile=hasproperty(first,:compile_time) ? first.compile_time : NaN
    println(io,join((name,T,n,Threads.nthreads(),checks.error,checks.products,first.time,compile,
                    BenchmarkTools.median(trial).time/1e9,minimum(trial).time/1e9,
                    trial.memory,trial.allocs,length(trial)),','))
    flush(io)
end
open(joinpath(out,"interval-spectral.csv"),"w") do io
    println(io,"operation,type,n,threads,reference_error,matrix_products,first_s,compile_s,median_s,min_s,bytes,allocations,samples")
    for T in (Float32,Float64,ComplexF32,ComplexF64), n in (32,96)
        R=real(T); low,high=R(3.07),R(4.43); center=(low+high)/2
        dense=Matrix(LA.Tridiagonal(fill(T(-1),n-1),fill(T(4),n),fill(T(-1),n-1)))
        A=CSRMatrix(dense); values=R[4-2cospi(k/(n+1)) for k in 1:n]
        selected=findall(x->low<=x<=high,values); expected=values[selected]
        limit=1000n*eps(R)
        interval_measure(io,:gershgorin,T,n,()->hermitian_spectral_bounds(A),
                         result->(;ok=result[1]<=values[1] && result[2]>=values[end],error=0.0,products=0))
        interval_measure(io,:plan_filter,T,n,()->polynomial_filter(A,low,high;degree=24,bounds=(2,6)),
                         P->(;ok=all(isfinite,P.coefficients),error=0.0,products=0))
        P=polynomial_filter(A,low,high;degree=24,bounds=(2,6)); workspace=FilterWorkspace(P;columns=4)
        X=randn(T,n,4); Y=similar(X)
        Q=T[sqrt(2/(n+1))*sinpi(i*j/(n+1)) for i in 1:n,j in 1:n]
        weights=[sum(P.coefficients[k+1]*cos(k*acos(clamp((x/P.scale-P.center)/P.radius,-one(R),one(R))))
                     for k in 0:length(P.coefficients)-1) for x in values]
        reference=Q*LA.Diagonal(weights)*Q'*X
        function apply_check(result)
            error=LA.norm(result-reference)/max(LA.norm(reference),eps(R))
            (;ok=error<=limit,error,products=(length(P.coefficients)-1)*size(X,2))
        end
        interval_measure(io,:apply_filter_workspace,T,n,()->apply_filter!(Y,P,X;workspace),apply_check)
        interval_measure(io,:sturm_count,T,n,()->count_eigenvalues_in_interval(A,low,high),
                         result->(;ok=result.count==length(expected) && result.complete,error=abs(result.count-length(expected)),products=0))
        function interval_check(result)
            error=length(result.values)==length(expected) ? LA.norm(result.values-expected)/max(LA.norm(expected),eps(R)) : Inf
            (;ok=result.converged && result.complete && error<=limit,error,products=result.matrix_products)
        end
        interval_measure(io,:tridiagonal_interval,T,n,()->eigenvalues_in_interval(A,low,high),interval_check)
        interval_measure(io,:full_krylov_interval,T,n,
                         ()->eigenvalues_in_interval(A,low,high;method=:krylov,subspace=n,maxblocks=n,blocksize=4),interval_check)
        wanted=min(3,length(expected)); target=sort(expected[sortperm(abs.(expected.-center))[1:wanted]])
        function filtered_check(result)
            error=length(result.values)==wanted ? LA.norm(result.values-target)/max(LA.norm(target),eps(R)) : Inf
            residual=result.count==0 ? Inf : maximum(result.residuals)
            (;ok=result.converged && error<=limit && residual<=10sqrt(eps(R))*max(one(R),maximum(abs,target)),
              error,products=result.matrix_products)
        end
        interval_measure(io,:filtered_interval,T,n,
                         ()->polynomial_filtered_eigenvalues(A,low,high,wanted;degree=24,bounds=(2,6),
                                                              subspace=min(n,32),keep=6,blocksize=3,maxiter=30),filtered_check)
    end
end
