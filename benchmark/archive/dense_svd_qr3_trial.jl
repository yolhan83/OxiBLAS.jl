# Experimental pivoted QR-LQ-QR preconditioning. Process-local candidate only.
using OxiBLAS, Random, Statistics
import LinearAlgebra as LA
LA.BLAS.set_num_threads(1)
const O=OxiBLAS
function upper_r(Q,n,::Type{T}) where T
    R=zeros(T,n,n)
    @inbounds for j in 1:n,i in 1:j
        R[i,j]=Q.factors[i,j]
    end
    R
end
function qr3_trial(A)
    T=eltype(A);m,n=size(A);C,scale=O._normalized_copy(A)
    norms=[nrm2(view(C,:,j)) for j in 1:n]
    @inbounds for j in 1:n
        iszero(norms[j]) && return svd_factor(A;threaded=false),true
        for i in 1:m
            C[i,j]=O._realdiv(C[i,j],norms[j])
        end
    end
    P=qr_pivoted(C;threaded=false);Q1=P.qr
    O._svd_dense_qr_eligible(Q1) || return svd_factor(A;threaded=false),true
    R1=upper_r(Q1,n,T)
    @inbounds for j in 1:n,i in 1:j
        R1[i,j]*=norms[P.p[j]]
    end
    Q2=qr_factor(Matrix(adjoint(R1));threaded=false)
    O._svd_dense_qr_eligible(Q2) || return svd_factor(A;threaded=false),true
    Q3=qr_factor(Matrix(adjoint(upper_r(Q2,n,T)));threaded=false)
    O._svd_dense_qr_eligible(Q3) || return svd_factor(A;threaded=false),true
    G=svd_factor(upper_r(Q3,n,T);threaded=false)
    apply_q!(G.U,Q3;threaded=false)
    F=O._svd_dense_qr_result(Q1,G,scale,m,n,false,false)
    apply_q!(F.Vt,Q2;side=:R,trans=:C,threaded=false)
    Vt=similar(F.Vt)
    @inbounds for j in 1:n,i in 1:n
        Vt[i,P.p[j]]=F.Vt[i,j]
    end
    O.SVDFactor(F.U,F.S,Vt,F.sweeps),false
end
for T in (Float32,Float64,ComplexF32,ComplexF64),n in parse.(Int,split(get(ENV,"OXIBLAS_SVD_QR3_SIZES","192,512"),',')),m in (n,3n)
    A=randn(MersenneTwister(0x5ad+13n+(T<:Complex ? 1 : 0)),T,m,n)
    for mode in (:current,:qr3)
        f=mode===:current ? ()->(svd_factor(A;threaded=false),false) : ()->qr3_trial(A)
        f();samples=[@timed(f()) for _ in 1:3];F,fallback=last(samples).value
        R=real(T);k=min(m,n)
        residual=LA.norm(A-F.U*LA.Diagonal(F.S)*F.Vt)/LA.norm(A)
        orth=max(LA.norm(adjoint(F.U)*F.U-Matrix{T}(LA.I,k,k)),LA.norm(F.Vt*adjoint(F.Vt)-Matrix{T}(LA.I,k,k)))/sqrt(k)
        spectrum=LA.norm(F.S-LA.svdvals(A))/LA.norm(F.S)
        max(residual,orth,spectrum)<=128eps(R)*max(m,n) || error("QR3 numerical guard failed")
        println("QR3 ",T," ",m," ",n," ",mode," seconds=",median(getproperty.(samples,:time)),
                " bytes=",minimum(getproperty.(samples,:bytes))," fallback=",fallback,
                " sweeps=",F.sweeps," residual=",residual," orth=",orth," spectrum=",spectrum)
        flush(stdout)
    end
end
