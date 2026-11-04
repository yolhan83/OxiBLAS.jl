using OxiBLAS, Random, Statistics
import LinearAlgebra as LA
LA.BLAS.set_num_threads(1)
const O=OxiBLAS
for T in (Float32,Float64,ComplexF32,ComplexF64), n in (192,512), m in (n,3n)
    A=randn(MersenneTwister(0x1ab),T,m,n);R=real(T)
    for block in (0,16,32,64)
        f=block==0 ? ()->O._svd_dense_bidiagonal_factor(A;blocked=false,threaded=false) :
                     ()->O._svd_dense_bidiagonal_blocked(A;blocksize=block,threaded=false)
        f();times=Float64[];F=nothing;bytes=typemax(Int)
        for _ in 1:5
            result=@timed f();F=result.value;push!(times,result.time);bytes=min(bytes,result.bytes)
        end
        # Independent random-action residual, outside timing, without building
        # square m-by-m U (which would hide the reduction's memory advantage).
        Z=randn(MersenneTwister(0x2ab),T,n,3);D=zeros(T,n,3)
        for j in 1:3,i in 1:n
            D[i,j]=F.factors[i,i]*Z[i,j]+(i<n ? F.factors[i,i+1]*Z[i+1,j] : zero(T))
        end
        P=O._svd_dense_backtransform(F,Z,false;threaded=false)
        Q=O._svd_dense_backtransform(F,D,true;threaded=false)
        residual=LA.norm((A/F.scale)*P-Q)/max(LA.norm(A/F.scale)*LA.norm(Z),floatmin(R))
        residual<=128eps(R)*max(m,n) || error("panel action residual $residual")
        println("REDUCTION ",T," ",m," ",n," block=",block," seconds=",median(times)," bytes=",bytes," residual=",residual)
        flush(stdout)
    end
end
