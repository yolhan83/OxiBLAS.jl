@testset "Sparse LU zero threshold preserves a usable pivot" begin
    for T in TYPES, threshold in (0, eps(real(T)), 1)
        for A in (T[0 1; 2 3], T[0 0 1; 1 2 3; 0 4 5])
            F=sparse_lu(CSRMatrix(A);pivot_threshold=threshold)
            @test all(isfinite,F.L.values) && all(isfinite,F.U.values)
            @test Matrix(F.L)*Matrix(F.U)≈A[F.p,F.q] rtol=tol(T) atol=tol(T)
            b=T.(1:size(A,1))
            @test A*solve(F,b)≈b rtol=5tol(T) atol=5tol(T)
        end
    end
    for T in TYPES
        @test_throws SingularException sparse_lu(CSRMatrix(T[0 1;0 2]);pivot_threshold=0)
        A=T[1 2;4 9]
        F=sparse_lu(CSRMatrix(A);pivot_threshold=0,ordering=:natural)
        @test F.p==[1,2]
        @test Matrix(F.L)*Matrix(F.U)≈A rtol=tol(T) atol=tol(T)
    end
end
