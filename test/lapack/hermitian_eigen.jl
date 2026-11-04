@testset "Symmetric/Hermitian tridiagonalization and eigenpairs" begin
    for T in TYPES, n in (0,1,2,3,9,33,65), uplo in (:U,:L)
        M=randn(T,n,n); A=Matrix(LA.Hermitian(M,uplo)); input=copy(M)
        # The ignored triangle is deliberately poisoned; diagonal imaginary
        # components also do not belong to the Hermitian input definition.
        for j in 1:n, i in 1:n
            if uplo===:U ? i>j : i<j
                input[i,j]=T(NaN)
            elseif i==j && T<:Complex
                input[i,j]=complex(real(input[i,j]),real(T)(NaN))
            end
        end
        F=tridiagonal_factor(input;uplo); Q=F.Q; H=F.T
        bound=tol(T)*max(1,LA.norm(A))*20
        @test LA.norm(A-Q*H*Q')<=bound
        @test LA.norm(Q'*Q-Matrix{T}(LA.I,n,n))<=tol(T)*max(1,n)*10
        @test all(F.offdiagonal.>=0)
        @test F.diagonal==LA.diag(H)
        @test F.offdiagonal==LA.diag(H,-1)
        for algorithm in (:unblocked,:blocked)
            @test tridiagonal_q(F;algorithm,blocksize=7)≈Q rtol=tol(T)*10 atol=tol(T)*10
        end
        for vectors in (true,false)
            E=eigen_hermitian(input;uplo,vectors)
            @test issorted(E.values)
            @test E.values≈LA.eigvals(LA.Hermitian(A)) rtol=tol(T)*20 atol=tol(T)*max(1,n)*20
            @test eltype(E.values)===real(T)
            if vectors
                @test LA.norm(A*E.vectors-E.vectors*LA.Diagonal(E.values))<=bound
                @test LA.norm(E.vectors'*E.vectors-Matrix{T}(LA.I,n,n))<=tol(T)*max(1,n)*10
                @test eltype(E.vectors)===T
            else
                @test E.vectors===nothing
            end
        end
    end
    for T in TYPES
        # Purely imaginary subdiagonal phases are essential: omitting D gives
        # plausible eigenvalues but wrong complex eigenvectors.
        raw=T<:Complex ? ComplexF64[2 1+2im 0;1-2im -1 3im;0 -3im 4] : [2.0 1 0;1 -1 3;0 3 4]
        for A in (T.(raw),zeros(T,9,9),T(3)*Matrix{T}(LA.I,9,9))
            E=eigen_hermitian(A)
            @test A*E.vectors≈E.vectors*LA.Diagonal(E.values) rtol=tol(T)*30 atol=tol(T)*30
            @test E.vectors'*E.vectors≈Matrix{T}(LA.I,size(A)...) rtol=tol(T)*30 atol=tol(T)*30
        end
        for exponent in (real(T)===Float32 ? (-140,-120,120) : (-1060,-1000,1000))
            scale=ldexp(one(real(T)),exponent); M=randn(T,7,7); M=M+M'; A=M*scale
            E=eigen_hermitian(A)
            @test all(isfinite,E.values)
            # Subnormal input/output rounding is bounded in units of the input
            # scale, instead of demanding impossible full relative precision.
            under=eps(zero(real(T)))/scale
            @test LA.norm((A/scale)*E.vectors-E.vectors*LA.Diagonal(E.values/scale))<=max(tol(T)*100,under*100)*LA.norm(M)
        end
    end
    @test_throws DimensionMismatch eigen_hermitian(ones(3,4))
    @test_throws ArgumentError eigen_hermitian([1.0 Inf;0 1])
    @test_throws ArgumentError eigen_hermitian(ones(3,3);uplo=:bad)
    @test_throws OxiBLAS.ConvergenceError eigen_hermitian(randn(7,7);maxiter=0)
end

@testset "Tridiagonal implicit QR" begin
    for R in (Float32,Float64), n in (0,1,2,3,9,33,65), vectors in (true,false)
        d=randn(R,n); e=randn(R,max(0,n-1)); saved=copy(d); saved_e=copy(e)
        E=eigen_tridiagonal(d,e;vectors)
        A=Matrix(LA.SymTridiagonal(d,e))
        @test d==saved && e==saved_e
        reference=n==0 ? R[] : LA.eigvals(LA.SymTridiagonal(d,e))
        @test E.values≈reference rtol=tol(R)*20 atol=tol(R)*max(1,n)*20
        if vectors
            @test LA.norm(A*E.vectors-E.vectors*LA.Diagonal(E.values))<=tol(R)*max(1,LA.norm(A))*20
            @test E.vectors'*E.vectors≈Matrix{R}(LA.I,n,n) rtol=tol(R)*30 atol=tol(R)*30
        else
            @test E.vectors===nothing
        end
    end
    for R in (Float32,Float64), d in (zeros(R,9),ones(R,9)), e in (zeros(R,8),ones(R,8))
        E=eigen_tridiagonal(d,e)
        @test E.values≈LA.eigvals(LA.SymTridiagonal(d,e)) rtol=tol(R)*20 atol=tol(R)*20
    end
    @test_throws DimensionMismatch eigen_tridiagonal(ones(3),ones(3))
    @test_throws ArgumentError eigen_tridiagonal([1.0,NaN],[0.1])
    @test_throws OxiBLAS.ConvergenceError eigen_tridiagonal(ones(9),ones(8);maxiter=0)
end
