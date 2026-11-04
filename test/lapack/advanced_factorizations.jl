@testset "v0.6 advanced native factorizations" begin
    for T in TYPES
        for dims in ((17,11),(11,17),(0,3),(3,0)), pivot in (:complete,:rook)
            A=randn(T,dims); F=lu_factor(A;pivot)
            @test F.L*F.U≈A[F.p,F.q] rtol=10tol(T) atol=10tol(T)
        end
        A=randn(T,19,19)+T(10)*LA.I; b=randn(T,19,3)
        for pivot in (:complete,:rook)
            F=lu_factor(A;pivot); @test A*solve(F,b)≈b rtol=10tol(T) atol=10tol(T)
        end
        for dims in ((35,19),(19,35),(0,3),(3,0)), bs in (1,3,8,32)
            M=randn(T,dims); F=qr_factor(M;blocksize=bs)
            @test F.Q*F.R≈M rtol=15tol(T) atol=15tol(T)
            @test F.Q'*F.Q≈Matrix{T}(LA.I,dims[1],dims[1]) rtol=15tol(T) atol=15tol(T)
        end
        M=randn(T,31,17); F=qr_pivoted(M)
        @test F.Q*F.R≈M[:,F.p] rtol=15tol(T) atol=15tol(T)
        rhs=randn(T,31); @test least_squares(F,rhs)≈M\rhs rtol=20tol(T) atol=20tol(T)
        M[:,17]=M[:,1]+M[:,2]; F=qr_factor(M;pivot=true)
        X=least_squares(F,rhs); @test M'*(M*X-rhs)≈zeros(T,17) atol=100tol(T)
        for herm in (false,true), uplo in (:L,:U)
            M=randn(T,19,19); M=M+(herm ? adjoint(M) : transpose(M))
            M+=T(3)*LA.I; F=ldlt_factor(M;hermitian=herm,uplo)
            @test F.L*F.D*(herm ? adjoint(F.L) : transpose(F.L))≈M[F.p,F.p] rtol=30tol(T) atol=30tol(T)
            rhs=randn(T,19,2); @test M*solve(F,rhs)≈rhs rtol=50tol(T) atol=50tol(T)
        end
        M=T[0 2;2 0]; F=bunchkaufman_factor(M)
        @test F.blocks==[2]
        @test M*solve(F,T[3,5])≈T[3,5] rtol=tol(T)
        @test_throws SingularException ldlt_factor(zeros(T,3,3))
        X=randn(T,17,17); B=X*X'+T(3)*LA.I
        X=randn(T,17,17); M=X+X'; E=eigen_generalized(M,B)
        @test M*E.vectors≈B*E.vectors*LA.Diagonal(E.values) rtol=80tol(T) atol=80tol(T)
        @test E.vectors'*B*E.vectors≈Matrix{T}(LA.I,17,17) rtol=80tol(T) atol=80tol(T)
        @test E.values≈LA.eigvals(LA.Hermitian(M),LA.Hermitian(B)) rtol=80tol(T) atol=80tol(T)
        rhs=randn(T,19,2); result=expert_solve(A,rhs;rtol=40eps(real(T)))
        @test result.converged
        @test A*result.x≈rhs rtol=15tol(T) atol=15tol(T)
        for dims in ((13,7),(7,13),(7,7),(0,3),(3,0))
            M=randn(T,dims); F=bidiagonal_factor(M)
            @test F.U*F.B*F.V'≈M rtol=20tol(T) atol=20tol(T)
            @test F.U'*F.U≈Matrix{T}(LA.I,dims[1],dims[1]) rtol=20tol(T) atol=20tol(T)
        end
    end
    for T in (Float64,ComplexF64)
        A=randn(T,23,23)+T(8)*LA.I; b=randn(T,23)
        R=mixed_solve(A,b)
        @test R.converged
        @test A*R.x≈b rtol=2e-12 atol=2e-12
    end
    for T in (ComplexF32,ComplexF64)
        A=randn(T,13,13); F=schur_factor(A); G=schur_reorder(F,z->real(z)>0)
        @test G.Z*G.T*G.Z'≈A rtol=100tol(T) atol=100tol(T)
        selected=real.(G.values).>0
        @test issorted(selected;rev=true)
        E=eigen_general(A)
        @test A*E.vectors≈E.vectors*LA.Diagonal(E.values) rtol=100tol(T) atol=100tol(T)
    end
end
