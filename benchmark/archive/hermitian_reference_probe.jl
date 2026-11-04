# Dedicated contract probe, separate from the real-diagonal timing workload.
using OxiBLAS, Random, InteractiveUtils
import LinearAlgebra as LA
LA.BLAS.set_num_threads(1)
const backend=get(ENV,"OXIBLAS_REFERENCE","openblas")
if backend=="accelerate"
    @eval using AppleAccelerate
    AppleAccelerate.set_num_threads(1)
end
Random.seed!(1)
open(joinpath(get(ENV,"OXIBLAS_REPORT_DIR",joinpath(@__DIR__,"..","..","reports")),"hermitian-contract-$(backend).csv"),"w") do io
    println(io,"backend,operation,type,uplo,trans,relative_error,max_imaginary_diagonal,passed")
    for T in (ComplexF32,ComplexF64), uplo in ('U','L'), trans in ('N','C'), rank2 in (false,true)
        n,k=9,7
        A=randn(T,trans=='N' ? (n,k) : (k,n)); B=randn(T,size(A)); C=randn(T,n,n)
        aa=trans=='N' ? A : A'; bb=trans=='N' ? B : B'
        α=rank2 ? T(.7+.2im) : T(.7); β=real(T)(-.3)
        expected=(rank2 ? α*aa*bb'+conj(α)*bb*aa' : real(α)*aa*aa')+β*Matrix(LA.Hermitian(C,uplo=='U' ? :U : :L))
        X=copy(C)
        rank2 ? LA.BLAS.her2k!(uplo,trans,α,A,B,β,X) : LA.BLAS.herk!(uplo,trans,real(α),A,β,X)
        selected=[(i,j) for j in 1:n for i in 1:n if (uplo=='U' ? i<=j : i>=j)]
        err=sqrt(sum(abs2(X[i,j]-expected[i,j]) for (i,j) in selected))/LA.norm(expected)
        imax=maximum(abs(imag(X[i,i])) for i in 1:n)
        passed=err<100eps(real(T))*n && iszero(imax)
        println(io,join((backend,rank2 ? :her2k : :herk,T,uplo,trans,err,imax,passed),','))
        Y=copy(C)
        rank2 ? her2k!(Y,A,B,α,β;uplo=Symbol(uplo),trans=Symbol(trans)) : herk!(Y,A,real(α),β;uplo=Symbol(uplo),trans=Symbol(trans))
        all(isapprox(Y[i,j],expected[i,j];rtol=100eps(real(T))*n,atol=100eps(real(T))*n) for (i,j) in selected) || error("native contract failure")
        all(iszero(imag(Y[i,i])) for i in 1:n) || error("native imaginary diagonal")
    end
end
