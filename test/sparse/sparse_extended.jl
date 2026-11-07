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

@testset "Additive Schwarz, SPAI and Block-GMRES" begin
    for T in TYPES
        n=18
        D=Matrix(LA.Tridiagonal(fill(T(-1),n-1),fill(T(5),n),fill(T(-1),n-1)))
        for i in 1:n-2
            v=T <: Real ? T(0.25) : T(0.25,0.1)
            D[i,i+2]=v; D[i+2,i]=conj(v)
        end
        C=CSRMatrix(D); b=randn(T,n); B=randn(T,n,3)

        # Single subdomain with exact solver (:lu, :cholesky) is the exact inverse.
        for sub in (:lu,:cholesky)
            P1=additive_schwarz(C;subdomains=1,overlap=0,solver=sub,threaded=false)
            z1=similar(b); OxiBLAS._precondition!(z1,b,P1)
            r1=similar(b); sparse_mul!(r1,C,z1)
            @test r1≈b rtol=20tol(T) atol=20tol(T)
        end

        # Overlapping subdomains: sequential vs threaded, aliasing, and concurrent calls.
        for sub in (:lu,:cholesky,:ilu0)
            Ps=additive_schwarz(C;subdomains=3,overlap=1,solver=sub,threaded=false)
            Pt=additive_schwarz(C;subdomains=3,overlap=1,solver=sub,threaded=true)
            zs=similar(b); OxiBLAS._precondition!(zs,b,Ps)
            zt=similar(b); OxiBLAS._precondition!(zt,b,Pt)
            @test zs≈zt rtol=10tol(T) atol=10tol(T)
            alias=copy(b); OxiBLAS._precondition!(alias,alias,Pt)
            @test alias≈zt rtol=10tol(T) atol=10tol(T)
            outs=[similar(b) for _ in 1:8]
            @sync for k in 1:8
                Threads.@spawn OxiBLAS._precondition!(outs[k],b,Pt)
            end
            for k in 1:8
                @test outs[k]≈zt rtol=10tol(T) atol=10tol(T)
            end
            if sub===:lu
                g=gmres(C,b;preconditioner=Pt,maxiter=n,rtol=20eps(real(T)))
                @test g.converged
                r=similar(b); sparse_mul!(r,C,g.x)
                @test r≈b rtol=30tol(T) atol=30tol(T)
            end
        end

        # SPAI: sequential vs threaded agreement, exact on diagonal, and preconditioner action.
        Diag=CSRMatrix(Matrix(LA.Diagonal(T[T(2+i) for i in 1:8])))
        Md=spai(Diag;pattern_levels=0,threaded=true)
        rd=randn(T,8); zd=similar(rd); OxiBLAS._precondition!(zd,rd,Md)
        checkd=similar(rd); sparse_mul!(checkd,Diag,zd)
        @test checkd≈rd rtol=10tol(T) atol=10tol(T)
        for lvl in (0,1)
            Ms=spai(C;pattern_levels=lvl,threaded=false)
            Mt=spai(C;pattern_levels=lvl,threaded=true)
            @test Matrix(Ms.inverse)≈Matrix(Mt.inverse) rtol=10tol(T) atol=10tol(T)
            zm=similar(b); OxiBLAS._precondition!(zm,b,Mt)
            refm=similar(b); sparse_mul!(refm,Mt.inverse,b)
            @test zm≈refm rtol=10tol(T) atol=10tol(T)
        end

        # Block-GMRES: unpreconditioned, preconditioned, and restarted multi-RHS solves.
        P_schwarz=additive_schwarz(C;subdomains=3,overlap=1,solver=:lu,threaded=true)
        cases=T===Float64 ? ((nothing,10),(P_schwarz,4),(spai(C;pattern_levels=1,threaded=true),4)) : ((P_schwarz,4),)
        for (prec,rest) in cases
            res=block_gmres(C,B;preconditioner=prec,restart=rest,maxiter=40,rtol=30eps(real(T)))
            @test res.converged
            AX=similar(B); sparse_mul!(AX,C,res.x)
            @test AX≈B rtol=50tol(T) atol=50tol(T)
            @test [LA.norm(view(AX,:,j)-view(B,:,j)) for j in axes(B,2)]≈res.residuals atol=20tol(T)
        end
    end
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

    # Spectral tuning format persistence and autotune check
    scfg=SpectralConfig(:golub_kahan,:multishift,8,192)
    sfixture=SpectralTuning(scfg,"Float64",16,OxiBLAS._tuning_cpu(),Dict{Symbol,Float64}(),Dict{Symbol,Float64}())
    mktemp() do path,io
        close(io); save_spectral_tuning(path,sfixture)
        sloaded=load_spectral_tuning(Float64,path)
        @test sloaded.svd_algorithm===:golub_kahan && sloaded.schur_algorithm===:multishift
        @test sloaded.svd_bidiag_threshold==8 && sloaded.schur_multishift_threshold==192
        @test_throws ArgumentError load_spectral_tuning(Float32,path)
    end

    # Explicit autotune execution on small size
    tune = autotune_spectral(Float32, 16; repeats=1)
    @test tune.config.svd_algorithm in (:golub_kahan, :jacobi, :divide_conquer)
    @test tune.config.schur_algorithm in (:standard, :multishift)
    OxiBLAS.clear_spectral_tuning!()
end
