# Only the new high-level SVD wrapper, not the other factorization families.
using Test, Random, OxiBLAS
import LinearAlgebra as LA
@testset "Dense SVD compatibility with current high-level API" begin
    rng=MersenneTwister(0x5ad2026)
    for T in (Float32,Float64,ComplexF32,ComplexF64), (m,n) in ((13,7),(7,13))
        R=real(T);A=randn(rng,T,m,n);old=copy(A);F=OxiBLAS.svd(A)
        @test F isa LA.Factorization
        @test size(F)==size(A)
        @test A==old
        U,S,V=F;bound=128eps(R)*max(m,n)
        @test LA.norm(A-U*LA.Diagonal(S)*adjoint(V))<=bound*LA.norm(A)
        @test OxiBLAS.svdvals(A)≈S rtol=bound
        @test F.engine isa OxiBLAS.SVDFactor
        b=randn(rng,T,m);x=F\b;reference=LA.pinv(A)*b
        @test x≈reference rtol=bound atol=bound
    end
end
