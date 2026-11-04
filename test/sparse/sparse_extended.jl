@testset "v0.6 sparse formats, products and iterative algorithms" begin
    for T in TYPES
        A=randn(T,29,29); A[abs.(A).<1].=zero(T); A+=T(20)*LA.I
        C=CSRMatrix(A); S=CSCMatrix(C); O=COOMatrix(C)
        @test Matrix(C)≈A
        @test Matrix(S)≈A
        @test Matrix(O)≈A
        @test Matrix(CSRMatrix(S))≈A
        @test Matrix(CSCMatrix(O))≈A
        coo=COOMatrix(3,4,[1,1,2,3],[2,2,1,4],T[1,2,0,5])
        @test length(coo.values)==2
        @test Matrix(CSRMatrix(coo))==T[0 3 0 0;0 0 0 0;0 0 0 5]
        for F in (C,S,O), tr in (:N,:T,:C)
            x=randn(T,29); y=fill(T(NaN),29)
            sparse_mul!(y,F,x;trans=tr,parallel_threshold=0)
            @test y≈op(A,tr)*x rtol=10tol(T) atol=10tol(T)
            X=randn(T,29,7); Y=fill(T(NaN),29,7)
            sparse_mul!(Y,F,X;trans=tr,parallel_threshold=0)
            @test Y≈op(A,tr)*X rtol=10tol(T) atol=10tol(T)
        end
        B=randn(T,29,17); B[abs.(B).<1].=zero(T)
        @test Matrix(sparse_matmul(C,CSRMatrix(B)))≈A*B rtol=10tol(T) atol=10tol(T)
        @test Matrix(sparse_matmul(S,COOMatrix(CSRMatrix(B))))≈A*B rtol=10tol(T) atol=10tol(T)
        b=randn(T,29); M=ilu0(C)
        for solver in (gmres,bicgstab), preconditioner in (nothing,M,[A[i,i] for i in 1:29])
            result=solver(C,b;preconditioner,maxiter=100,rtol=20eps(real(T)))
            @test result.converged
            @test LA.norm(A*result.x-b)≈result.residual atol=10tol(T)
            @test A*result.x≈b rtol=20tol(T) atol=20tol(T)
        end
        # ILU(0) is exact on this tridiagonal structure.
        D=Matrix(LA.Tridiagonal(fill(T(-1),9),fill(T(4),10),fill(T(-1),9)))
        M=ilu0(CSRMatrix(D)); rhs=randn(T,10)
        @test D*solve(M,rhs)≈rhs rtol=10tol(T) atol=10tol(T)
        H=randn(T,13,13); H=H+H'
        result=lanczos(CSRMatrix(H);nev=3,maxiter=13,rtol=50eps(real(T)))
        @test result.values≈sort(LA.eigvals(LA.Hermitian(H));rev=true)[1:3] rtol=30tol(T) atol=30tol(T)
        @test H*result.vectors≈result.vectors*LA.Diagonal(result.values) rtol=30tol(T) atol=30tol(T)
        @test result.converged
        limited=gmres(C,b;maxiter=0)
        @test !limited.converged && limited.reason===:maxiter
        @test limited.residual≈LA.norm(b)
    end
    @test_throws ArgumentError COOMatrix(2,2,[3],[1],[1.0])
    @test_throws ArgumentError CSCMatrix(2,2,[1,2,3],[1,3],[1.0,2.0])
    @test_throws SingularException ilu0(CSRMatrix(zeros(3,3)))
    @test_throws ArgumentError gmres(CSRMatrix([2.0 0;0 2]),ones(2);preconditioner=zeros(2))
end

@testset "Tuning persistence without running calibration" begin
    cfg=GemmConfig(Float64;mc=96,kc=128,nc=256)
    # A format fixture, not a measured performance result.
    fixture=GemmTuning(cfg,:scalar,"Float64",(3,4,5),OxiBLAS._tuning_cpu(),Float64[],GemmConfig[])
    mktemp() do path,io
        close(io); save_gemm_tuning(path,fixture)
        loaded=load_gemm_tuning(Float64,path)
        @test loaded.config.mc==96 && loaded.config.kc==128 && loaded.config.nc==256
        @test loaded.dimensions==(3,4,5) && loaded.kernel===:scalar
        @test_throws ArgumentError load_gemm_tuning(Float32,path)
    end
end
