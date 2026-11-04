@testset "Compact Hessenberg reduction" begin
    for T in TYPES, n in (0,1,2,3,7,19,65)
        A=randn(T,n,n); saved=copy(A); F=hessenberg_factor(A)
        for algorithm in (:unblocked,:blocked)
            Q=hessenberg_q(F;algorithm,blocksize=7); H=F.H
            @test A==saved
            @test LA.norm(A-Q*H*Q')<=tol(T)*max(1,LA.norm(A))*10
            @test LA.norm(Q'*Q-Matrix{T}(LA.I,n,n))<=tol(T)*max(1,n)*10
            @test all(iszero(H[i,j]) for j in 1:n for i in j+2:n)
        end
        @test hasproperty(F,:H) && hasproperty(F,:Q)
        if n<=19
            Q=F.Q
            for side in (:L,:R), trans in (:N,:T,:C), algorithm in (:unblocked,:blocked)
                B=randn(T,side===:L ? (n,9) : (9,n)); old=copy(B)
                expected=side===:L ? op(Q,trans)*old : old*op(Q,trans)
                @test apply_hessenberg_q!(B,F;side,trans,algorithm,blocksize=3)===B
                @test B≈expected rtol=tol(T)*10 atol=tol(T)*10
            end
            x=randn(T,n); old=copy(x)
            @test apply_hessenberg_q!(x,F)≈Q*old rtol=tol(T)*10 atol=tol(T)*10
        end
    end
    for T in TYPES, n in (3,17), exponent in (real(T)===Float32 ? (-120,120) : (-1000,1000))
        scale=ldexp(one(real(T)),exponent); A=randn(T,n,n)*scale
        F=hessenberg_factor(A); Q=F.Q; H=F.H
        @test all(isfinite,H)
        @test LA.norm(A/scale-Q*(H/scale)*Q')<=tol(T)*100*LA.norm(A/scale)
    end
    for T in TYPES
        A=randn(T,16,16); viewA=view(A,16:-2:2,15:-2:1); old=copy(viewA)
        F=hessenberg_factor(viewA)
        @test viewA==old
        @test F.Q*F.H*F.Q'≈old rtol=tol(T)*10 atol=tol(T)*10
        # Snapshot the compact factors before overwriting them through B.
        F=hessenberg_factor(randn(T,9,9)); Q=F.Q; expected=Q*copy(F.factors)
        @test apply_hessenberg_q!(F.factors,F;algorithm=:blocked,blocksize=3)≈expected rtol=tol(T)*10 atol=tol(T)*10
        # Already Hessenberg, diagonal, zero and block diagonal inputs.
        for A in (zeros(T,9,9),Matrix{T}(LA.I,9,9),Matrix(LA.UpperTriangular(randn(T,9,9))))
            F=hessenberg_factor(A)
            @test F.H≈A
            @test F.Q==Matrix{T}(LA.I,9,9)
        end
    end
    @test_throws DimensionMismatch hessenberg_factor(ones(3,4))
    @test_throws ArgumentError hessenberg_factor([1.0 Inf;0 1])
    F=hessenberg_factor(randn(7,7))
    @test size(apply_hessenberg_q!(zeros(7,0),F;algorithm=:blocked))==(7,0)
    @test size(apply_hessenberg_q!(zeros(0,7),F;side=:R,algorithm=:blocked))==(0,7)
    @test_throws DimensionMismatch apply_hessenberg_q!(ones(6,3),F)
    @test_throws ArgumentError apply_hessenberg_q!(ones(7),F;side=:R)
    @test_throws ArgumentError hessenberg_q(F;blocksize=0)
    @test_throws ArgumentError hessenberg_q(F;algorithm=:bad)
end
