# Prepared only. Invoked by the supervised launcher after explicit approval.
using OxiBLAS, BenchmarkTools
import LinearAlgebra as LA
length(ARGS)==1 || error("usage: sparse_svd_memory.jl NEW_OUTPUT_DIRECTORY")
out=abspath(ARGS[1]); ispath(out) && error("refusing to overwrite results"); mkpath(out)
open(joinpath(out,"sparse-svd-memory.csv"),"w") do io
    println(io,"operation,type,m,n,threads,Julia_ms,Rust_ms,OpenBLAS_ms,Accelerate_ms,allocated_bytes,allocations,retained_bytes,workspace_payload_bytes,operator_matvecs,verification_matvecs,left_residual,right_residual")
    for T in (Float32,Float64,ComplexF32,ComplexF64), n in (256,4096,16384), wide in (false,true)
        R=real(T); m,p=wide ? (n,4n) : (4n,n); vals=Vector{T}(undef,n)
        for i in 1:n
            s=i<=n-4 ? one(R)+R(i)/R(n) : R(12+4*(i-(n-3)))
            vals[i]=T<:Complex ? s*T(cis(R(i)/7)) : T(s)
        end
        A=CSRMatrix(COOMatrix(m,p,collect(1:n),collect(1:n),vals)); rank=4; cap=24
        w=SparseSVDWorkspace(A,rank;subspace=cap)
        for operation in (:golub_kahan_fresh,:golub_kahan_reuse,:dilation)
            f=operation===:dilation ? ()->sparse_svd_lanczos(A,rank;algorithm=:dilation,subspace=cap,keep=8) :
                ()->sparse_svd_lanczos(A,rank;subspace=cap,keep=8,workspace=operation===:golub_kahan_reuse ? w : nothing)
            first=f(); expected=R[24,20,16,12]
            first.converged && LA.norm(first.S-expected)<=1000sqrt(eps(R))*LA.norm(expected) || error("sparse memory guard failed")
            left=maximum(first.left_residuals); right=maximum(first.right_residuals)
            trial=run(@benchmarkable($f(),evals=1);seconds=0.15,samples=3)
            println(io,join((operation,T,m,p,Threads.nthreads(),BenchmarkTools.median(trial).time/1e6,"N/A","N/A","N/A",
                trial.memory,trial.allocs,Base.summarysize(first),operation===:golub_kahan_reuse ? sparse_svd_workspace_bytes(w) : 0,
                first.operator_matvecs,first.verification_matvecs,left,right),','));flush(io)
        end
    end
end
