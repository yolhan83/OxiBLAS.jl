@testset "v0.6 task-owned contexts and complex compact SIMD" begin
    for T in TYPES
        cfg=GemmConfig(T;mc=16,kc=13,nc=16,parallel_threshold=0)
        w=GemmWorkspace(T,67,67,43;config=cfg)
        A=randn(T,67,43); B=randn(T,43,67); C=zeros(T,67,67)
        gemmt!(C,A,B;workspace=w,config=cfg,blocksize=16)
        children=copy(w.children); tiles=copy(w.tiles); jobs=w.jobs
        gemmt!(C,A,B;workspace=w,config=cfg,blocksize=16)
        @test all(w.children[i]===children[i] for i in eachindex(children))
        @test all(w.tiles[i]===tiles[i] for i in eachindex(tiles))
        @test w.jobs===jobs
        aa=[randn(T,19,17) for _ in 1:9]; bb=[randn(T,17,23) for _ in 1:9]
        cc=[zeros(T,19,23) for _ in 1:9]; dd=deepcopy(cc)
        bw=GemmWorkspace(T,19,23,17;config=cfg)
        @sync begin
            Threads.@spawn gemm_batched!(cc,aa,bb;workspace=bw,config=cfg)
            Threads.@spawn gemm_batched!(dd,aa,bb;workspace=bw,config=cfg)
        end
        @test all(isapprox(cc[i],aa[i]*bb[i];rtol=8tol(T),atol=8tol(T)) for i in eachindex(cc))
        @test all(isapprox(cc[i],dd[i];rtol=8tol(T)) for i in eachindex(cc))
        @test length(bw.children)==min(Threads.nthreads(),9)
        GC.gc() # exercise buffers retained by cached child contexts
        gemm_batched!(dd,aa,bb;workspace=bw,config=cfg)
        @test all(isapprox(cc[i],dd[i];rtol=8tol(T)) for i in eachindex(cc))
        for side in (:L,:R), uplo in (:L,:U), tr in (:N,:T,:C), unit in (false,true)
            # Off-diagonal entries scaled by 1/N: a unit triangle ignores the +20I,
            # and with O(1) random entries its condition number (hence the residual
            # of the trsm check) depended on the random draw.
            N=33; M=17; tri=randn(T,N,N)/N; tri+=T(20)*LA.I
            triangular=uplo===:U ? LA.UpperTriangular(tri) : LA.LowerTriangular(tri)
            unit && (triangular=uplo===:U ? LA.UnitUpperTriangular(tri) : LA.UnitLowerTriangular(tri))
            Z=randn(T,side===:L ? (N,M) : (M,N)); old=copy(Z); R=op(triangular,tr)
            trmm!(Z,tri,T(.7);side,uplo,trans=tr,unit,config=cfg)
            @test Z≈(side===:L ? T(.7)*R*old : T(.7)*old*R) rtol=8tol(T) atol=8tol(T)
            copyto!(Z,old); trsm!(Z,tri,T(.7);side,uplo,trans=tr,unit,config=cfg)
            @test (side===:L ? R*Z : Z*R)≈T(.7)*old rtol=8tol(T) atol=8tol(T)
        end
        for n in (1,3,9,65), uplo in (:L,:U), herm in (false,true)
            M=randn(T,n,n); P=pack_matrix(M;uplo,kind=herm ? :hermitian : :symmetric)
            x=randn(T,n); y=randn(T,n); expected=Matrix(P)*x+T(.2)*y
            f=herm ? hpmv! : spmv!
            f(y,P,x,one(T),T(.2);parallel_threshold=0)
            @test y≈expected rtol=8tol(T) atol=8tol(T)
            scalar=zeros(T,n); vector=similar(scalar)
            f(scalar,P,x;kernel=:scalar,threaded=false)
            f(vector,P,x;threaded=false)
            @test vector≈scalar rtol=8tol(T) atol=8tol(T)
        end
        for tr in (:N,:T,:C)
            P=band_matrix(randn(T,65,47);kl=7,ku=3)
            x=randn(T,tr===:N ? 47 : 65); y=zeros(T,tr===:N ? 65 : 47)
            gbmv!(y,P,x;trans=tr,parallel_threshold=0)
            @test y≈op(Matrix(P),tr)*x rtol=8tol(T) atol=8tol(T)
            M=randn(T,65,47); expected=op(M,tr)*x
            gemv!(y,M,x;trans=tr,parallel_threshold=0)
            @test y≈expected rtol=8tol(T) atol=8tol(T)
        end
    end
    unchanged=fill(7.0,3,4,0)
    @test_throws DimensionMismatch gemm_strided_batched!(unchanged,zeros(3,2,0),zeros(5,4,0))
    C=fill(7.0,3,3)
    @test_throws ArgumentError gemm_batched!([C],[ones(3,3)],[ones(3,3)];kernel=:invalid)
    @test all(==(7),C)
end

@testset "Small packing-free products and ComplexF32 K tails" begin
    for T in TYPES, m in (1,3,4,7,8), n in (1,5,8), k in (1,3,7,129,512)
        A=randn(T,m,k); B=randn(T,k,n); C=fill(T(NaN),m,n)
        gemm!(C,A,B)
        @test C≈A*B rtol=12tol(T) atol=12tol(T)
    end
    for k in 1:17
        A=randn(ComplexF32,35,k); B=randn(ComplexF32,k,29); C=zeros(ComplexF32,35,29)
        gemm!(C,A,B;algorithm=:simd)
        @test C≈A*B rtol=10tol(ComplexF32) atol=10tol(ComplexF32)
    end
end
