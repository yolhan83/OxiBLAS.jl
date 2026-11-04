# spectrum_error: test/helpers/checks.jl

@testset "Real/complex implicit Schur QR" begin
    for T in TYPES, n in (0,1,2,3,7,19,41,65), seed in 1:3
        A=randn(T,n,n); old=copy(A); F=schur_factor(A); Z,H=F.Z,F.T
        bound=tol(T)*max(1,nrm2(A))*20
        @test A==old
        @test LA.norm(A-Z*H*Z')<=bound
        @test LA.norm(Z'*Z-Matrix{T}(LA.I,n,n))<=tol(T)*max(1,n)*10
        @test all(iszero(H[i,j]) for j in 1:n for i in j+(T<:Real ? 2 : 1):n)
        @test spectrum_error(F.values,LA.eigvals(A))<=bound*10
        if T<:Real
            @test eltype(F.T)===T
            @test all(iszero(H[i+1,i]) || iszero(H[i+2,i+1]) for i in 1:n-2)
            for i in 1:n-1
                if !iszero(H[i+1,i])
                    @test abs(H[i,i]-H[i+1,i+1])<=bound
                    @test H[i,i+1]*H[i+1,i]<0
                end
            end
        end
    end
    for T in TYPES
        cases=([0 1;-1 0],[0 1;1 0],[1 1;0 1],[0 0;1 0],
               [0 1 0;1 0 1;0 1 0],[0 1 0;0 0 1;1 0 0],
               zeros(9,9),Matrix{Float64}(LA.I,9,9))
        for original in cases
            A=T.(original); F=schur_factor(A)
            @test F.Z*F.T*F.Z'≈A rtol=tol(T)*30 atol=tol(T)*30
            @test spectrum_error(F.values,LA.eigvals(A))<=tol(T)*100
        end
        # Non-normal Jordan chains: residual/unitarity remain meaningful even
        # when individual eigenvalues are severely ill-conditioned.
        A=zeros(T,17,17)
        for i in 1:17
            A[i,i]=T(2)
            i<17 && (A[i,i+1]=one(T))
        end
        F=schur_factor(A;maxiter=0)
        @test F.iterations==0
        @test F.values==fill(T(2),17)
        @test F.Z*F.T*F.Z'≈A
        for exponent in (real(T)===Float32 ? (-120,120) : (-1000,1000))
            scale=ldexp(one(real(T)),exponent); M=randn(T,17,17); A=M*scale
            F=schur_factor(A)
            @test all(isfinite,F.T)
            @test LA.norm(M-F.Z*(F.T/scale)*F.Z')<=tol(T)*100*LA.norm(M)
            @test spectrum_error(F.values/scale,LA.eigvals(M))<=tol(T)*300*LA.norm(M)
        end
        A0=randn(T,19,19); A=view(A0,19:-2:1,18:-2:2)
        # A deliberately nonsquare view is rejected before reduction.
        @test_throws DimensionMismatch schur_factor(A)
        V=view(A0,19:-2:1,19:-2:1); F=schur_factor(V)
        @test F.Z*F.T*F.Z'≈V rtol=tol(T)*50 atol=tol(T)*50
    end
    @test_throws OxiBLAS.ConvergenceError schur_factor(randn(7,7);maxiter=0)
    @test_throws OxiBLAS.ConvergenceError schur_factor(randn(ComplexF64,7,7);maxiter=0)
    @test_throws ArgumentError schur_factor(ones(3,3);maxiter=-1)
    @test_throws ArgumentError schur_factor([NaN 0.0;0 1])
end
