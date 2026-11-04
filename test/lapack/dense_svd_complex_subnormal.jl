using Test, OxiBLAS
import LinearAlgebra as LA
@testset "Unit singular-vector phases for complex subnormal entries" begin
    for T in (ComplexF32,ComplexF64),offdiag in (false,true)
        R=real(T);tiny=nextfloat(zero(R));A=zeros(T,3,3)
        A[1,1]=one(T);A[2,2]=complex(tiny,tiny)
        offdiag && (A[2,3]=complex(tiny,-tiny))
        F=svd_bidiagonal(A)
        @test F.S[2]>0
        @test LA.norm(adjoint(F.U)*F.U-Matrix{T}(LA.I,3,3))<=128eps(R)
        @test LA.norm(F.Vt*adjoint(F.Vt)-Matrix{T}(LA.I,3,3))<=128eps(R)
    end
end
