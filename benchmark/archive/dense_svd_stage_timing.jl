using OxiBLAS, Random
const O=OxiBLAS
import LinearAlgebra as LA
LA.BLAS.set_num_threads(1)
med(f;k=7)=(f(); t=[(@elapsed f()) for _ in 1:k]; sort!(t)[(k+1)÷2]*1e3)
for T in (Float32,Float64,ComplexF64), (m,n) in ((64,64),(192,192),(576,192))
    A=randn(MersenneTwister(1),T,m,n)
    tot=med(()->svd_bidiagonal(A;threaded=false))
    vals=med(()->svd_bidiagonal(A;vectors=false,threaded=false))
    lap=med(()->LA.svd(A))
    B=m>=2n ? O._svd_dense_tall_qr(A,false,false)[2] : A
    tqr=m>=2n ? med(()->O._svd_dense_tall_qr(A,false,false)) : 0.0
    red=med(()->O._svd_dense_bidiagonal_factor(B;threaded=false))
    F=O._svd_dense_bidiagonal_factor(B;threaded=false)
    d,e,l,r=O._real_bidiagonal(F); R=real(T); k=size(B,2)
    qrv=med(()->(dd=copy(d);ee=copy(e);O._bidiagonal_qr!(dd,ee,O._identity(R,k),O._identity(R,k),10^7,8eps(R))))
    qrn=med(()->(dd=copy(d);ee=copy(e);O._bidiagonal_qr!(dd,ee,nothing,nothing,10^7,8eps(R))))
    X=T.(O._identity(R,k))
    bt=med(()->(O._svd_dense_backtransform(F,X,true);O._svd_dense_backtransform(F,X,false)))
    println(rpad("$T $m×$n",22)," total=",round(tot,digits=2)," vals=",round(vals,digits=2)," LAPACKsvd=",round(lap,digits=2),
      " | tallqr=",round(tqr,digits=2)," red=",round(red,digits=2)," bdsqr+vec=",round(qrv,digits=2)," bdsqr=",round(qrn,digits=2)," backtr=",round(bt,digits=2))
end
