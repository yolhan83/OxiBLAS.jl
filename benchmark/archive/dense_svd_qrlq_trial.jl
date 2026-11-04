# Candidate only, never included by src/. Native QR-LQ preconditioning of
# Jacobi; reject weak pivots before composing any transformed singular vectors.
using OxiBLAS, Random, Statistics
import LinearAlgebra as LA
LA.BLAS.set_num_threads(1)
const O=OxiBLAS
function qrlq_trial(A)
    T=eltype(A);m,n=size(A)
    Q1,R1,scale=O._svd_dense_tall_qr(A,true,false)
    O._svd_dense_qr_eligible(Q1) || return (svd_factor(A;threaded=false),true)
    Q2=qr_factor(Matrix(adjoint(R1));threaded=false)
    O._svd_dense_qr_eligible(Q2) || return (svd_factor(A;threaded=false),true)
    R2=zeros(T,n,n)
    @inbounds for j in 1:n,i in 1:j
        R2[i,j]=Q2.factors[i,j]
    end
    G=svd_factor(Matrix(adjoint(R2));threaded=false)
    F=O._svd_dense_qr_result(Q1,G,scale,m,n,false,false)
    apply_q!(F.Vt,Q2;side=:R,trans=:C,threaded=false)
    F,false
end
for T in (Float32,Float64,ComplexF32,ComplexF64),n in parse.(Int,split(get(ENV,"OXIBLAS_SVD_QRLQ_SIZES","192"),',')),m in (n,3n)
    A=randn(MersenneTwister(0x5ad+13n+(T<:Complex ? 1 : 0)),T,m,n)
    for mode in (:current,:qrlq)
        f=mode===:current ? ()->(svd_factor(A;threaded=false),false) : ()->qrlq_trial(A)
        f();samples=[@timed(f()) for _ in 1:5];F,fallback=last(samples).value
        R=real(T);k=min(m,n)
        residual=LA.norm(A-F.U*LA.Diagonal(F.S)*F.Vt)/LA.norm(A)
        orth=max(LA.norm(adjoint(F.U)*F.U-Matrix{T}(LA.I,k,k)),LA.norm(F.Vt*adjoint(F.Vt)-Matrix{T}(LA.I,k,k)))/sqrt(k)
        spectrum=LA.norm(F.S-LA.svdvals(A))/LA.norm(F.S)
        max(residual,orth,spectrum)<=128eps(R)*max(m,n) || error("QR-LQ numerical guard failed")
        println("QRLQ ",T," ",m," ",n," ",mode," seconds=",median(getproperty.(samples,:time)),
                " bytes=",minimum(getproperty.(samples,:bytes))," fallback=",fallback,
                " sweeps=",F.sweeps," residual=",residual," orth=",orth," spectrum=",spectrum)
        flush(stdout)
    end
end
