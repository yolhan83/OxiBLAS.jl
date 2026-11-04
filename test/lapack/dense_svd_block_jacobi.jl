using Test, Random, OxiBLAS
import LinearAlgebra as LA
@testset "Blocked one-sided Jacobi with robust final convergence" begin
    rng=MersenneTwister(0xb10c)
    for T in (Float32,Float64,ComplexF32,ComplexF64)
        R=real(T)
        for (m,n) in ((192,192),(288,128),(128,288),(130,130)), algorithm in (:jacobi,:parallel_jacobi,:block_jacobi,:parallel_block_jacobi)
            A=randn(rng,T,m,n);old=copy(A);k=min(m,n)
            F=svd_factor(A;algorithm,threaded=true)
            rtol=8eps(R)*sqrt(R(k));G=F.U'*F.U-Matrix{T}(LA.I,k,k)
            @test A==old
            @test LA.norm(A-F.U*LA.Diagonal(F.S)*F.Vt)<=128eps(R)*max(m,n)*LA.norm(A)
            @test maximum(abs,G)<=2rtol
            @test LA.norm(F.Vt*F.Vt'-Matrix{T}(LA.I,k,k))<=128eps(R)*max(m,n)*sqrt(k)
            @test F.S≈LA.svdvals(A) rtol=128eps(R)*max(m,n)
            values=svd_factor(A;algorithm,vectors=false,threaded=true)
            @test values.U===nothing && values.Vt===nothing
            @test values.S≈F.S rtol=128eps(R)*max(m,n)
        end
        for A in (ones(T,144,128),randn(rng,T,144,4)*randn(rng,T,4,128))
            F=svd_factor(A;algorithm=:parallel_jacobi)
            @test LA.norm(A-F.U*LA.Diagonal(F.S)*F.Vt)<=128eps(R)*144*LA.norm(A)
            @test LA.norm(F.U'*F.U-Matrix{T}(LA.I,128,128))<=128eps(R)*144*sqrt(128)
        end
        powers=round.(Int,range(0,R===Float32 ? -100 : -900;length=128))
        d=R[ldexp(one(R),p) for p in powers];A=Matrix{T}(LA.Diagonal(d))
        # Mix only adjacent magnitudes: a known graded spectrum with no
        # representational loss of the smallest singular values in the input.
        c=R(0.8);s=R(0.6)
        for i in 1:2:127
            OxiBLAS._svd_rotate_columns!(A,i,i+1,c,T(s))
        end
        for algorithm in (:jacobi,:parallel_jacobi,:block_jacobi,:parallel_block_jacobi)
            F=svd_factor(A;algorithm)
            @test all(F.S.>0)
            @test maximum(abs.(F.S./d.-1))<=512eps(R)
        end
    end
end
