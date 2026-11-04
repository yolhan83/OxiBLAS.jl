using OxiBLAS, Random, Statistics, Profile
const O=OxiBLAS
import LinearAlgebra as LA
LA.BLAS.set_num_threads(1)
function stages(A)
    t=@elapsed F=O._svd_dense_bidiagonal_factor(A;threaded=false)
    d,e,l,r=O._real_bidiagonal(F);n=length(d);R=eltype(d);T=eltype(A)
    U=O._identity(R,n);V=O._identity(R,n)
    q=@elapsed O._bidiagonal_qr!(d,e,U,V,160n*n,8eps(R))
    X=T.(U);Y=T.(V)
    for j in 1:n,i in 1:n
        X[i,j]*=l[i];Y[i,j]*=r[i]
    end
    u=@elapsed O._svd_dense_backtransform(F,X,true;threaded=false)
    v=@elapsed O._svd_dense_backtransform(F,Y,false;threaded=false)
    (reduction=t,qr=q,left=u,right=v)
end
for T in (Float32,Float64,ComplexF32,ComplexF64)
    A=randn(MersenneTwister(0x5ad),T,576,192)
    stages(A);rs=[stages(A) for _ in 1:7]
    println(T," ",map(k->k=>median(getproperty.(rs,k)),keys(first(rs))))
end
A=randn(MersenneTwister(0x5ad),ComplexF32,576,192)
svd_factor(A;threaded=false);Profile.clear()
@profile for _ in 1:12
    svd_factor(A;threaded=false)
end
Profile.print(format=:flat,mincount=5,sortedby=:count)
