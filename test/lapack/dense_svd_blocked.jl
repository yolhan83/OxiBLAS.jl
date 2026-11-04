# Standalone algorithm tests, also includable from the dense campaign runner.
using Test, Random, OxiBLAS
import LinearAlgebra as LA
@testset "LABRD panels and delayed two-sided updates" begin
    rng=MersenneTwister(0x1ab)
    for T in (Float32,Float64,ComplexF32,ComplexF64), (m,n) in ((19,11),(39,33),(151,96)), nb in (1,7,32)
        A=randn(rng,T,m,n);old=copy(A)
        F=OxiBLAS._svd_dense_bidiagonal_blocked(A;blocksize=nb,threaded=true)
        bound=64eps(real(T))*max(m,n)
        @test A==old
        @test LA.norm(A-F.U*F.B*F.V')<=bound*LA.norm(A)
        @test LA.norm(F.U'*F.U-Matrix{T}(LA.I,m,m))<=bound*sqrt(m)
        @test LA.norm(F.V'*F.V-Matrix{T}(LA.I,n,n))<=bound*sqrt(n)
    end
    for T in (Float32,Float64,ComplexF32,ComplexF64)
        R=real(T);m,n=151,96
        cases=(zeros(T,m,n),ones(T,m,n),randn(rng,T,m,3)*randn(rng,T,3,n),
               Matrix{T}(LA.Diagonal([ldexp(one(R),-i) for i in 0:n-1])))
        for A in cases
            F=svd_bidiagonal(A;threaded=true);k=min(size(A)...)
            bound=128eps(R)*max(size(A)...)
            @test LA.norm(A-F.U*LA.Diagonal(F.S)*F.Vt)<=bound*max(one(R),LA.norm(A))
            @test LA.norm(F.U'*F.U-Matrix{T}(LA.I,k,k))<=bound*sqrt(k)
            @test LA.norm(F.Vt*F.Vt'-Matrix{T}(LA.I,k,k))<=bound*sqrt(k)
            @test F.S≈LA.svdvals(A) rtol=bound atol=bound*max(one(R),LA.norm(A))
        end
    end
end

@testset "Representable QR shifts below the squared exponent range" begin
    for T in (Float32,Float64,ComplexF32,ComplexF64)
        R=real(T);scale=R===Float32 ? R(1e-25) : R(1e-200)
        B=Matrix{T}(LA.Bidiagonal(R[1,2,3,4],R[0.5,0.5,0.5],:U))
        A=zeros(T,5,5);A[1,1]=one(T);A[2:5,2:5]=B*scale
        F=svd_bidiagonal(A)
        @test F.S[2:5]/scale≈LA.svdvals(B) rtol=256eps(R)
        @test LA.norm(F.U'*F.U-Matrix{T}(LA.I,5,5))<512eps(R)
        @test LA.norm(F.Vt*F.Vt'-Matrix{T}(LA.I,5,5))<512eps(R)
    end
end
