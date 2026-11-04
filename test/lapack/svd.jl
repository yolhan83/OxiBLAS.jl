@testset "Native real/complex one-sided Jacobi SVD" begin
    for T in TYPES, (m,n) in ((0,0),(3,0),(0,3),(1,1),(1,7),(7,1),(3,3),(13,7),(7,13),(33,29)), vectors in (true,false)
        A=randn(T,m,n); old=copy(A); F=svd_factor(A;vectors); k=min(m,n)
        @test A==old
        @test length(F.S)==k && issorted(F.S;rev=true) && all(F.S.>=0)
        @test F.S≈LA.svdvals(A) rtol=tol(T)*20 atol=tol(T)*max(1,LA.norm(A))*20
        if vectors
            @test size(F.U)==(m,k) && size(F.Vt)==(k,n)
            @test LA.norm(A-F.U*LA.Diagonal(F.S)*F.Vt)<=tol(T)*max(1,LA.norm(A))*30
            @test F.U'*F.U≈Matrix{T}(LA.I,k,k) rtol=tol(T)*30 atol=tol(T)*30
            @test F.Vt*F.Vt'≈Matrix{T}(LA.I,k,k) rtol=tol(T)*30 atol=tol(T)*30
        else
            @test F.U===nothing && F.Vt===nothing
        end
    end
    for T in TYPES, (m,n) in ((9,7),(7,9))
        for A in (zeros(T,m,n),ones(T,m,n),[T(i*j) for i in 1:m,j in 1:n])
            F=svd_factor(A)
            @test F.U*LA.Diagonal(F.S)*F.Vt≈A rtol=tol(T)*100 atol=tol(T)*100
            @test F.U'*F.U≈Matrix{T}(LA.I,min(m,n),min(m,n)) rtol=tol(T)*50 atol=tol(T)*50
            @test F.Vt*F.Vt'≈Matrix{T}(LA.I,min(m,n),min(m,n)) rtol=tol(T)*50 atol=tol(T)*50
        end
        A=randn(T,m,3)*randn(T,3,n); F=svd_factor(A)
        @test F.U*LA.Diagonal(F.S)*F.Vt≈A rtol=tol(T)*100 atol=tol(T)*100
        @test F.U'*F.U≈Matrix{T}(LA.I,min(m,n),min(m,n)) rtol=tol(T)*50 atol=tol(T)*50
    end
    for T in TYPES
        for exponent in (real(T)===Float32 ? (-140,-120,120) : (-1060,-1000,1000))
            scale=ldexp(one(real(T)),exponent); M=randn(T,9,7); A=M*scale; F=svd_factor(A)
            under=eps(zero(real(T)))/scale
            @test all(isfinite,F.S)
            @test LA.norm(A/scale-F.U*LA.Diagonal(F.S/scale)*F.Vt)<=max(tol(T)*100,under*100)*LA.norm(M)
        end
        # A truly graded diagonal must not be truncated using an absolute eps
        # threshold, or by squaring very small column norms.
        powers=real(T)===Float32 ? [0,-20,-60,-100] : [0,-100,-500,-900]
        d=real(T)[ldexp(one(real(T)),p) for p in powers]
        A=Matrix{T}(LA.Diagonal(d)); F=svd_factor(A;maxsweeps=0)
        @test F.S≈d rtol=tol(T) atol=0
        @test F.U*LA.Diagonal(F.S)*F.Vt≈A
        M=randn(T,19,17); V=view(M,19:-2:1,17:-2:1); F=svd_factor(V)
        @test F.U*LA.Diagonal(F.S)*F.Vt≈V rtol=tol(T)*50 atol=tol(T)*50
    end
    @test_throws OxiBLAS.ConvergenceError svd_factor(ones(3,2);maxsweeps=0)
    @test_throws ArgumentError svd_factor(ones(3,2);maxsweeps=-1)
    @test_throws ArgumentError svd_factor(ones(3,2);rtol=0)
    @test_throws ArgumentError svd_factor([1.0 Inf;0 1])
end
