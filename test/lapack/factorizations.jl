@testset "Blocked LU and solves" begin
    for T in TYPES, (m,n) in ((0,0),(1,1),(9,9),(13,7),(7,13),(67,65),(65,67)), block in (1,7,32)
        A=randn(T,m,n)
        F=lu_factor(A;blocksize=block)
        @test F.L*F.U≈A[F.p,:] rtol=tol(T)*10 atol=tol(T)*max(1,n)
        if m==n
            b=randn(T,n,3); x=solve(F,b)
            @test A*x≈b rtol=tol(T)*10 atol=tol(T)*max(1,n)
            v=randn(T,n)
            @test A*solve(F,v)≈v rtol=tol(T)*10 atol=tol(T)*max(1,n)
        end
    end
    @test_throws OxiBLAS.SingularException lu_factor(zeros(3,3))
    F=lu_factor(zeros(3,3);check=false)
    @test F.info==1
    @test_throws OxiBLAS.SingularException solve(F,ones(3))
    @test_throws DimensionMismatch solve(lu_factor(randn(3,3)),ones(4))
end

@testset "Blocked real/complex Cholesky" begin
    for T in TYPES, n in (0,1,9,65,99), block in (1,7,32)
        M=randn(T,n,n); A=M*M'+T(n+1)*Matrix{T}(LA.I,n,n)
        F=cholesky_factor(A;blocksize=block)
        @test F.L*F.L'≈A rtol=tol(T)*10 atol=tol(T)*max(1,n)
        b=randn(T,n,3)
        @test A*solve(F,b)≈b rtol=tol(T)*10 atol=tol(T)*max(1,n)
        v=randn(T,n)
        @test A*solve(F,v)≈v rtol=tol(T)*10 atol=tol(T)*max(1,n)
        # Upper triangle is not part of the input contract.
        @inbounds for j in 1:n, i in 1:j-1
            A[i,j]=T(NaN)
        end
        @test cholesky_factor(A;blocksize=block).L≈F.L rtol=tol(T)*10
    end
    @test_throws OxiBLAS.PosDefException cholesky_factor([1.0 0;0 -1])
    @test_throws DimensionMismatch cholesky_factor(ones(3,4))
end

@testset "Householder QR and least squares" begin
    for T in TYPES, (m,n) in ((0,0),(3,0),(0,3),(1,1),(13,7),(7,13),(67,39))
        A=randn(T,m,n); F=qr_factor(A); Q=F.Q; R=F.R
        @test Q*R≈A rtol=tol(T)*10 atol=tol(T)*max(1,n)
        @test Q'*Q≈Matrix{T}(LA.I,m,m) rtol=tol(T)*10 atol=tol(T)*max(1,m)
        if m>=n
            b=randn(T,m,3); x=least_squares(F,b)
            @test LA.norm(A'*(A*x-b))<=tol(T)*100*max(1,LA.norm(A)*LA.norm(b))
            if n>0
                @test x≈A\b rtol=tol(T)*30 atol=tol(T)*30
            end
            v=randn(T,m)
            @test solve(F,v)≈vec(least_squares(F,reshape(v,m,1))) rtol=tol(T)
        end
    end
    @test_throws OxiBLAS.SingularException least_squares(ones(5,3),ones(5))
    @test_throws ArgumentError least_squares(randn(3,5),ones(3))
    @test_throws DimensionMismatch least_squares(randn(5,3),ones(4))
    for n in (1,17,67)
        A=randn(n,n)+n*Matrix{Float64}(LA.I,n,n); b=randn(n)
        result=mixed_solve(A,b)
        @test result.converged
        @test A*result.x≈b rtol=2e-12 atol=2e-12
        @test result.iterations<=10
    end
end
