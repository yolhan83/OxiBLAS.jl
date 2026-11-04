@testset "AINV biconjugation, unexecuted reference cases" begin
    for T in TYPES
        A=randn(T,7,7)+T(12)*LA.I
        P=ainv(CSRMatrix(A);droptol=0,maxnnz=7)
        @test Matrix(P.W)'*A*Matrix(P.Z)≈LA.Diagonal(P.diagonal) rtol=20tol(T) atol=20tol(T)
        b=randn(T,7); z=similar(b); OxiBLAS._precondition!(z,b,P)
        @test A*z≈b rtol=20tol(T) atol=20tol(T)
        alias=copy(b); OxiBLAS._precondition!(alias,alias,P)
        @test alias≈z rtol=tol(T) atol=tol(T)
        z2=similar(z)
        @sync begin
            Threads.@spawn OxiBLAS._precondition!(z,b,P)
            Threads.@spawn OxiBLAS._precondition!(z2,b,P)
        end
        @test z≈z2 rtol=tol(T)
        GC.gc(); OxiBLAS._precondition!(z,b,P)
        @test A*z≈b rtol=20tol(T) atol=20tol(T)
        Q=ainv(CSRMatrix(A);droptol=.1,maxnnz=2)
        @test all(count(!iszero,Matrix(Q.Z)[:,j])<=2 for j in 1:7)
        @test_throws SingularException ainv(CSRMatrix(zeros(T,2,2)))
        E=ainv(CSRMatrix(zeros(T,0,0))); @test isempty(E.diagonal)
    end
    @test_throws ArgumentError ainv(CSRMatrix(ones(2,2));maxnnz=0)
end
